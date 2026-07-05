import apiClient from './index';
import { toCamelCase } from './utils';

export type ResearchSourceStatus = 'available' | 'partial' | 'missing' | string;

export type ResearchDataSource = {
  name: string;
  files: number;
  rows: number;
  dateRange: string;
  status: ResearchSourceStatus;
  sizeMb?: number;
};

export type ResearchTrainingReadiness = {
  status: string;
  summary: string;
};

export type ResearchQlibProvider = {
  path: string;
  calendarDays: number;
  calendarRange: string;
  instruments: number;
  featureDirs?: number;
  sizeMb?: number;
};

export type ResearchTrainingResult = {
  status: string;
  experimentName?: string;
  runId?: string;
  provider?: string;
  providerUri?: string;
  qlibProvider?: string;
  sourceConfig?: string;
  runtimeConfig?: string;
  metrics?: Record<string, number>;
  artifacts?: string[];
  generatedAt?: string;
};

export type ResearchCashControlMetrics = {
  days?: number;
  cumulativeReturn?: number | null;
  annualizedReturn?: number | null;
  meanDailyReturn?: number | null;
  dailyVolatility?: number | null;
  sharpeRatio?: number | null;
  /** Legacy cash-control reports used this name for a zero-risk-rate Sharpe ratio. */
  informationRatio?: number | null;
  maxDrawdown?: number | null;
  winRate?: number | null;
  bestDay?: number | null;
  worstDay?: number | null;
};

export type ResearchCashControlBacktest = {
  generatedAt?: string;
  mode?: string;
  comparison?: {
    baseline?: ResearchCashControlMetrics;
    cashControl?: ResearchCashControlMetrics;
    delta?: ResearchCashControlMetrics;
  };
  signalSummary?: {
    days?: number;
    regimeCounts?: Record<string, number>;
    averageCashRatio?: number | null;
    averagePositionRatio?: number | null;
    latestRegime?: string | null;
    latestCashRatio?: number | null;
    latestPositionRatio?: number | null;
  };
  artifacts?: Record<string, string>;
};

export type ResearchQlibPredictionScore = {
  instrument: string;
  score?: number | null;
};

export type ResearchQlibPredictions = {
  generatedAt?: string;
  mode?: string;
  runId?: string;
  experimentName?: string;
  predictionPath?: string;
  latestDate?: string;
  scoreColumn?: string;
  topk?: number;
  top?: ResearchQlibPredictionScore[];
  bottom?: ResearchQlibPredictionScore[];
  summary?: {
    rows?: number;
    latestRows?: number;
    dateCount?: number;
    scoreMin?: number | null;
    scoreMax?: number | null;
    scoreMean?: number | null;
  };
  artifacts?: Record<string, string>;
};

export type ResearchExperimentRun = {
  runId?: string;
  experimentName?: string;
  status: string;
  model?: string;
  strategy?: string;
  datasetRange?: string;
  conclusion?: string;
  nextStep?: string;
  provider?: string;
  qlibProvider?: string;
  providerUri?: string;
  sourceConfig?: string;
  runtimeConfig?: string;
  metrics?: Record<string, number>;
  startedAt?: string;
  completedAt?: string;
  logPath?: string;
  configPath?: string;
  mlflowUrl?: string;
};

export type ResearchExperimentNotebook = {
  status: string;
  latestNote?: string;
  runs: ResearchExperimentRun[];
  totalRuns: number;
  invalidLines?: number;
  logPath?: string;
  runsPath?: string;
};

export type ResearchExternalTool = {
  name: string;
  status: string;
  installed: boolean;
  mode: string;
  path?: string | null;
  summary: string;
  capabilities: string[];
  restrictions: string[];
};

export type ResearchExternalTools = {
  tonghuashun?: ResearchExternalTool;
  iphoneMirroring?: ResearchExternalTool;
};

export type ResearchCapabilityMatrixItem = {
  id: string;
  module: string;
  status: string;
  entry: string;
  route?: string;
  tests: string;
  browser: string;
  review: string;
  next: string;
  restrictions?: string[];
};

export type ResearchFunctionInventoryItem = {
  id: string;
  area: string;
  name: string;
  status: string;
  entry: string;
  route?: string;
  owner: string;
  coverage: string;
  gaps: string;
  next: string;
  verification: string;
};

export type ResearchChinaAstockReuseItem = {
  id: string;
  capability: string;
  source: string;
  status: string;
  priority: string;
  decision: string;
  target: string;
  next: string;
  restrictions?: string[];
};

export type ResearchAcceptanceChecklistItem = {
  id: string;
  title: string;
  status: string;
  route?: string;
  evidence: string;
  next: string;
};

export type ResearchExperimentMutationResult = {
  removedRun: ResearchExperimentRun;
  noteRemoved?: boolean;
  experimentNotebook: ResearchExperimentNotebook;
};

export type ResearchStatus = {
  generatedAt?: string;
  trainable: boolean;
  trainingReadiness: ResearchTrainingReadiness;
  dataSources: ResearchDataSource[];
  qlibProvider?: ResearchQlibProvider | null;
  latestTraining?: ResearchTrainingResult | null;
  cashControlBacktest?: ResearchCashControlBacktest | null;
  qlibPredictions?: ResearchQlibPredictions | null;
  experimentNotebook?: ResearchExperimentNotebook | null;
  externalTools?: ResearchExternalTools | null;
  capabilityMatrix?: ResearchCapabilityMatrixItem[] | null;
  functionInventory?: ResearchFunctionInventoryItem[] | null;
  chinaAstockReuse?: ResearchChinaAstockReuseItem[] | null;
  acceptanceChecklist?: ResearchAcceptanceChecklistItem[] | null;
};

export const researchApi = {
  async getStatus(): Promise<ResearchStatus> {
    const response = await apiClient.get<Record<string, unknown>>('/api/v1/research/status');
    return toCamelCase<ResearchStatus>(response.data);
  },

  async archiveExperimentRun(runId: string): Promise<ResearchExperimentMutationResult> {
    const response = await apiClient.post<Record<string, unknown>>(
      `/api/v1/research/experiments/${encodeURIComponent(runId)}/archive`,
    );
    return toCamelCase<ResearchExperimentMutationResult>(response.data);
  },

  async permanentlyDeleteExperimentRun(runId: string): Promise<ResearchExperimentMutationResult> {
    const response = await apiClient.delete<Record<string, unknown>>(
      `/api/v1/research/experiments/${encodeURIComponent(runId)}/permanent`,
    );
    return toCamelCase<ResearchExperimentMutationResult>(response.data);
  },
};
