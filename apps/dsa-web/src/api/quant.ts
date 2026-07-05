import apiClient from './index';
import type {
  ResearchCashControlBacktest,
  ResearchDataSource,
  ResearchExperimentNotebook,
  ResearchQlibPredictions,
  ResearchQlibProvider,
  ResearchTrainingReadiness,
  ResearchTrainingResult,
} from './research';
import { toCamelCase } from './utils';

export type QuantJobStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled' | string;

export interface QuantJob {
  jobId: string;
  jobType: string;
  label: string;
  status: QuantJobStatus;
  params: Record<string, unknown>;
  command: string[];
  cwd: string;
  pid?: number | null;
  returnCode?: number | null;
  createdAt: string;
  startedAt?: string | null;
  completedAt?: string | null;
  updatedAt: string;
  error?: string | null;
  logPath: string;
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
}

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
  experimentNotebook?: ResearchExperimentNotebook | null;
  mlflow?: QuantMlflowStatus;
}

export interface QuantJobListResponse {
  total: number;
  items: QuantJob[];
}

export interface QuantJobLog {
  jobId: string;
  content: string;
  truncated: boolean;
  logPath: string;
}

function convertJob(data: unknown): QuantJob {
  return toCamelCase<QuantJob>(data);
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
    experimentNotebook: camel.experimentNotebook ?? null,
    mlflow: camel.mlflow,
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
    return toCamelCase<QuantJob>(response.data);
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
