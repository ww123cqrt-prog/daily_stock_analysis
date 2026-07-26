import apiClient from './index';
import {
  convertQlibResearch,
  type ResearchCashControlBacktest,
  type ResearchDataSource,
  type ResearchExperimentNotebook,
  type ResearchQlibPredictions,
  type ResearchQlibResearch,
  type ResearchQlibProvider,
  type ResearchRobustSignal,
  type ResearchRobustStrategy,
  type ResearchTrainingReadiness,
  type ResearchTrainingResult,
} from './research';
import { toCamelCase } from './utils';

export type QuantJobStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled' | string;

export interface QuantJob {
  jobId: string;
  jobType: string;
  label: string;
  status: QuantJobStatus;
  params: Record<string, unknown>;
  pid?: number | null;
  returnCode?: number | null;
  createdAt: string;
  startedAt?: string | null;
  completedAt?: string | null;
  updatedAt: string;
  error?: string | null;
  artifactPaths: string[];
}

export interface QuantJobTypeInfo {
  label: string;
  allowedParams: string[];
}

export interface QuantMlflowStatus {
  url: string;
  status: string;
  role: string;
  checkedAt?: string | null;
  healthChecked?: boolean;
  error?: string | null;
}

export type QuantExternalServiceStatus = QuantMlflowStatus;

export interface QuantStatus {
  availableJobTypes: Record<string, QuantJobTypeInfo>;
  activeCount: number;
  latestJobs: Record<string, QuantJob>;
  recentJobs: QuantJob[];
  generatedAt?: string;
  trainable: boolean;
  trainingReadiness?: ResearchTrainingReadiness;
  dataSources: ResearchDataSource[];
  qlibProvider?: ResearchQlibProvider | null;
  latestTraining?: ResearchTrainingResult | null;
  cashControlBacktest?: ResearchCashControlBacktest | null;
  qlibPredictions?: ResearchQlibPredictions | null;
  qlibResearch?: ResearchQlibResearch | null;
  robustStrategy?: ResearchRobustStrategy | null;
  robustSignal?: ResearchRobustSignal | null;
  experimentNotebook?: ResearchExperimentNotebook | null;
  mlflow?: QuantMlflowStatus;
  externalServices?: Record<string, QuantExternalServiceStatus>;
}

export interface QuantJobListResponse {
  total: number;
  items: QuantJob[];
}

export interface QuantJobLog {
  jobId: string;
  content: string;
  truncated: boolean;
}

function convertJob(data: unknown): QuantJob {
  const job = toCamelCase<QuantJob & { command?: unknown; cwd?: unknown; logPath?: unknown }>(data);
  delete job.command;
  delete job.cwd;
  delete job.logPath;
  return job;
}

function convertJobTypeInfo(data: unknown): QuantJobTypeInfo {
  return toCamelCase<QuantJobTypeInfo>(data);
}

function convertRecordValues<T>(data: unknown, converter: (value: unknown) => T): Record<string, T> {
  if (!data || typeof data !== 'object') {
    return {};
  }
  return Object.fromEntries(
    Object.entries(data as Record<string, unknown>).map(([key, value]) => [key, converter(value)]),
  );
}

function convertStatus(data: Record<string, unknown>): QuantStatus {
  const camel = toCamelCase<Partial<QuantStatus>>(data);
  return {
    availableJobTypes: convertRecordValues(data.available_job_types, convertJobTypeInfo),
    activeCount: Number(data.active_count || 0),
    latestJobs: convertRecordValues(data.latest_jobs, convertJob),
    recentJobs: Array.isArray(data.recent_jobs) ? data.recent_jobs.map(convertJob) : [],
    generatedAt: camel.generatedAt,
    trainable: Boolean(camel.trainable),
    trainingReadiness: camel.trainingReadiness,
    dataSources: Array.isArray(camel.dataSources) ? camel.dataSources : [],
    qlibProvider: camel.qlibProvider ?? null,
    latestTraining: camel.latestTraining ?? null,
    cashControlBacktest: camel.cashControlBacktest ?? null,
    qlibPredictions: camel.qlibPredictions ?? null,
    qlibResearch: convertQlibResearch(data.qlib_research),
    robustStrategy: camel.robustStrategy ?? null,
    robustSignal: camel.robustSignal ?? null,
    experimentNotebook: camel.experimentNotebook ?? null,
    mlflow: camel.mlflow,
    externalServices: camel.externalServices ?? {},
  };
}

export function parseQuantJobStreamEvent(data: string): QuantJob[] {
  try {
    const payload = JSON.parse(data) as { jobs?: unknown };
    return Array.isArray(payload.jobs) ? payload.jobs.map(convertJob) : [];
  } catch {
    return [];
  }
}

export const quantApi = {
  async getStatus(): Promise<QuantStatus> {
    const response = await apiClient.get<Record<string, unknown>>('/api/v1/quant/status');
    return convertStatus(response.data);
  },

  async submitJob(jobType: string, params: Record<string, unknown> = {}): Promise<QuantJob> {
    const response = await apiClient.post<Record<string, unknown>>('/api/v1/quant/jobs', {
      job_type: jobType,
      params,
    });
    return convertJob(response.data);
  },

  async listJobs(params: { limit?: number; jobType?: string } = {}): Promise<QuantJobListResponse> {
    const response = await apiClient.get<Record<string, unknown>>('/api/v1/quant/jobs', {
      params: {
        limit: params.limit,
        job_type: params.jobType,
      },
    });
    const data = response.data;
    return {
      total: Number(data.total || 0),
      items: Array.isArray(data.items) ? data.items.map(convertJob) : [],
    };
  },

  async getJob(jobId: string): Promise<QuantJob> {
    const response = await apiClient.get<Record<string, unknown>>(
      `/api/v1/quant/jobs/${encodeURIComponent(jobId)}`,
    );
    return convertJob(response.data);
  },

  async getJobLog(jobId: string): Promise<QuantJobLog> {
    const response = await apiClient.get<Record<string, unknown>>(
      `/api/v1/quant/jobs/${encodeURIComponent(jobId)}/log`,
    );
    return toCamelCase<QuantJobLog>(response.data);
  },

  async cancelJob(jobId: string): Promise<QuantJob> {
    const response = await apiClient.post<Record<string, unknown>>(
      `/api/v1/quant/jobs/${encodeURIComponent(jobId)}/cancel`,
    );
    return convertJob(response.data);
  },

  getJobArtifactUrl(jobId: string, artifactIndex: number): string {
    const baseUrl = apiClient.defaults.baseURL || '';
    return `${baseUrl}/api/v1/quant/jobs/${encodeURIComponent(jobId)}/artifacts/${artifactIndex}`;
  },

  async getJobArtifactText(jobId: string, artifactIndex: number): Promise<string> {
    const response = await apiClient.get<string>(
      `/api/v1/quant/jobs/${encodeURIComponent(jobId)}/artifacts/${artifactIndex}`,
      { responseType: 'text' },
    );
    return typeof response.data === 'string'
      ? response.data
      : JSON.stringify(response.data, null, 2);
  },

  getJobStreamUrl(): string {
    const baseUrl = apiClient.defaults.baseURL || '';
    return `${baseUrl}/api/v1/quant/jobs/stream`;
  },
};
