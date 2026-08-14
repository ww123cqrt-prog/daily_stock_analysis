/**
 * Quant Research API client.
 *
 * Provides typed access to the /api/v1/quant-research endpoints
 * for Research, Experiment, Run, and Algorithm Registry.
 */

import apiClient from './index';
import { toCamelCase } from './utils';

// === Types ===

export type ResearchStatus = 'draft' | 'active' | 'completed' | 'archived';
export type ExperimentStatus = 'draft' | 'submitted' | 'running' | 'completed' | 'failed' | 'cancelled';
export type ExperimentType = 'stock_selection' | 'strategy';
export type MethodKind = 'rule' | 'algorithm';

export interface AlgorithmCapability {
  supportsClassification: boolean;
  supportsRegression: boolean;
  supportsFeatureImportance: boolean;
  supportsOnlineLearning: boolean;
  requiresGpu: boolean;
  maxFeatures: number | null;
  supportedLossFunctions: string[];
}

export interface AlgorithmConfigSchema {
  parameters: Record<string, {
    type: string;
    enum?: unknown[];
    minimum?: number;
    maximum?: number;
    default?: unknown;
    description?: string;
  }>;
  required: string[];
}

export interface Algorithm {
  id: string;
  name: string;
  nameZh: string;
  category: string;
  description: string;
  descriptionZh: string;
  capability: AlgorithmCapability;
  configSchema: AlgorithmConfigSchema;
  defaultParams: Record<string, unknown>;
}

export interface Research {
  id: string;
  name: string;
  description: string;
  status: ResearchStatus;
  researchType: ExperimentType;
  config: Record<string, unknown>;
  createdAt: string;
  updatedAt: string;
  experiments?: Experiment[];
}

export interface Experiment {
  id: string;
  researchId: string;
  name: string;
  description: string;
  experimentType: ExperimentType;
  methodKind: MethodKind;
  algorithmId: string | null;
  protocol: Record<string, unknown>;
  protocolHash: string | null;
  status: ExperimentStatus;
  createdAt: string;
  updatedAt: string;
  runs?: Run[];
}

export interface Run {
  id: string;
  experimentId: string;
  runLabel: string;
  status: string;
  startedAt: string | null;
  completedAt: string | null;
  metrics: Record<string, unknown>;
  artifacts: unknown[];
  errorMessage: string | null;
  isLegacy: boolean;
  mlflowRunId: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface QuantResearchOverview {
  totalResearch: number;
  totalExperiments: number;
  totalRuns: number;
  recentResearch: Research[];
}

export interface CreateResearchRequest {
  name: string;
  description?: string;
  researchType?: ExperimentType;
  config?: Record<string, unknown>;
}

export interface UpdateResearchRequest {
  name?: string;
  description?: string;
  status?: ResearchStatus;
  researchType?: ExperimentType;
  config?: Record<string, unknown>;
}

export interface CreateExperimentRequest {
  name: string;
  description?: string;
  experimentType?: ExperimentType;
  methodKind?: MethodKind;
  algorithmId?: string;
  protocol?: Record<string, unknown>;
}

export interface UpdateExperimentRequest {
  name?: string;
  description?: string;
  experimentType?: ExperimentType;
  methodKind?: MethodKind;
  algorithmId?: string | null;
  protocol?: Record<string, unknown>;
  status?: ExperimentStatus;
}

export interface CreateRunRequest {
  runLabel?: string;
  isLegacy?: boolean;
  mlflowRunId?: string;
}

// === Conversion helpers ===

function convertResearch(data: unknown): Research {
  return toCamelCase<Research>(data);
}

function convertExperiment(data: unknown): Experiment {
  return toCamelCase<Experiment>(data);
}

function convertRun(data: unknown): Run {
  return toCamelCase<Run>(data);
}

function convertAlgorithm(data: unknown): Algorithm {
  return toCamelCase<Algorithm>(data);
}

// === API client ===

export const quantResearchApi = {
  // Overview
  async getOverview(): Promise<QuantResearchOverview> {
    const response = await apiClient.get('/api/v1/quant-research/overview');
    return toCamelCase<QuantResearchOverview>(response.data);
  },

  // Research CRUD
  async listResearch(limit = 50, offset = 0): Promise<Research[]> {
    const response = await apiClient.get('/api/v1/quant-research/research', {
      params: { limit, offset },
    });
    return (response.data as unknown[]).map(convertResearch);
  },

  async getResearch(researchId: string): Promise<Research> {
    const response = await apiClient.get(
      `/api/v1/quant-research/research/${encodeURIComponent(researchId)}`,
    );
    return convertResearch(response.data);
  },

  async createResearch(body: CreateResearchRequest): Promise<Research> {
    const response = await apiClient.post('/api/v1/quant-research/research', body);
    return convertResearch(response.data);
  },

  async updateResearch(researchId: string, body: UpdateResearchRequest): Promise<Research> {
    const response = await apiClient.patch(
      `/api/v1/quant-research/research/${encodeURIComponent(researchId)}`,
      body,
    );
    return convertResearch(response.data);
  },

  async deleteResearch(researchId: string): Promise<void> {
    await apiClient.delete(
      `/api/v1/quant-research/research/${encodeURIComponent(researchId)}`,
    );
  },

  // Experiment CRUD
  async listExperiments(researchId: string): Promise<Experiment[]> {
    const response = await apiClient.get(
      `/api/v1/quant-research/research/${encodeURIComponent(researchId)}/experiments`,
    );
    return (response.data as unknown[]).map(convertExperiment);
  },

  async getExperiment(experimentId: string): Promise<Experiment> {
    const response = await apiClient.get(
      `/api/v1/quant-research/experiments/${encodeURIComponent(experimentId)}`,
    );
    return convertExperiment(response.data);
  },

  async createExperiment(researchId: string, body: CreateExperimentRequest): Promise<Experiment> {
    const response = await apiClient.post(
      `/api/v1/quant-research/research/${encodeURIComponent(researchId)}/experiments`,
      body,
    );
    return convertExperiment(response.data);
  },

  async updateExperiment(experimentId: string, body: UpdateExperimentRequest): Promise<Experiment> {
    const response = await apiClient.patch(
      `/api/v1/quant-research/experiments/${encodeURIComponent(experimentId)}`,
      body,
    );
    return convertExperiment(response.data);
  },

  async deleteExperiment(experimentId: string): Promise<void> {
    await apiClient.delete(
      `/api/v1/quant-research/experiments/${encodeURIComponent(experimentId)}`,
    );
  },

  // Run CRUD
  async listRuns(experimentId: string): Promise<Run[]> {
    const response = await apiClient.get(
      `/api/v1/quant-research/experiments/${encodeURIComponent(experimentId)}/runs`,
    );
    return (response.data as unknown[]).map(convertRun);
  },

  async getRun(runId: string): Promise<Run> {
    const response = await apiClient.get(
      `/api/v1/quant-research/runs/${encodeURIComponent(runId)}`,
    );
    return convertRun(response.data);
  },

  async createRun(experimentId: string, body: CreateRunRequest = {}): Promise<Run> {
    const response = await apiClient.post(
      `/api/v1/quant-research/experiments/${encodeURIComponent(experimentId)}/runs`,
      body,
    );
    return convertRun(response.data);
  },

  // Algorithm Registry
  async listAlgorithms(category?: string): Promise<Algorithm[]> {
    const response = await apiClient.get('/api/v1/quant-research/algorithms', {
      params: category ? { category } : undefined,
    });
    return (response.data as unknown[]).map(convertAlgorithm);
  },

  async getAlgorithm(algorithmId: string): Promise<Algorithm> {
    const response = await apiClient.get(
      `/api/v1/quant-research/algorithms/${encodeURIComponent(algorithmId)}`,
    );
    return convertAlgorithm(response.data);
  },
};
