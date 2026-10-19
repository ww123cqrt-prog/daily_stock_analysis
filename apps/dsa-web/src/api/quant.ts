import apiClient from './index';
import {
  convertQlibResearch,
  type ResearchCashControlBacktest,
  type ResearchCurrentProviderIdentity,
  type ResearchDataSource,
  type ResearchExperimentNotebook,
  type ResearchQlibPredictions,
  type ResearchQlibResearch,
  type ResearchQlibProvider,
  type ResearchProviderQuality,
  type ResearchRobustSignal,
  type ResearchRobustStrategy,
  type ResearchTrainingReadiness,
  type ResearchTrainingPreflight,
  type ResearchTrainingResult,
} from './research';
import { toCamelCase } from './utils';

export type QuantJobStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled' | string;

export type LocalServiceName = 'catdesk' | 'mlflow';

export interface LocalServiceStatus {
  service: LocalServiceName;
  status: string;
  url: string;
  openUrl: string;
  error?: string | null;
  pid?: number | null;
  port?: number | null;
  message?: string | null;
}

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
  progress?: {
    stage?: string;
    completed?: number;
    total?: number;
    percent?: number;
    current?: string;
    currentArtifact?: string;
    elapsedSeconds?: number;
    etaSeconds?: number | null;
    updatedAt?: string;
    pauseState?: string;
    lifecycleStatus?: string;
    runId?: string;
    batchId?: string;
    reasonCode?: string;
    nextGate?: string;
  };
  runnerAlive?: boolean | null;
  canCancel?: boolean;
  logUpdatedAt?: string | null;
}

export interface QuantJobTypeInfo {
  label: string;
  allowedParams: string[];
}

export const QUANT_API_CONTRACT_VERSION = 'quant-workbench-v2';

export interface QuantApiContract {
  version: string;
  backendBuildId: string;
  backendStartedAt: string;
}

export interface QuantJobTypesCatalog {
  availableJobTypes: Record<string, QuantJobTypeInfo>;
  apiContract: QuantApiContract;
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
  apiContract?: QuantApiContract;
  availableJobTypes: Record<string, QuantJobTypeInfo>;
  activeCount: number;
  latestJobs: Record<string, QuantJob>;
  recentJobs: QuantJob[];
  generatedAt?: string;
  trainable: boolean;
  trainingReadiness?: ResearchTrainingReadiness;
  trainingPreflight?: ResearchTrainingPreflight;
  trainingPreflights?: Record<string, ResearchTrainingPreflight>;
  dataSources: ResearchDataSource[];
  qlibProvider?: ResearchQlibProvider | null;
  providerQuality?: Record<string, ResearchProviderQuality>;
  currentProviderIdentities?: Record<string, ResearchCurrentProviderIdentity>;
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

export interface QuantStatusOptions {
  includeResearch?: boolean;
}

export interface QuantDataStatusDataset {
  datasetId: string;
  snapshotId?: string;
  manifestSha256?: string;
  contentSha256?: string;
  latestTradeDate?: string;
  priceBasis?: string;
  dataKind?: string;
  role?: string;
  moduleReleaseSha256?: string;
  contentManifestSha256?: string;
  requestedThrough?: string;
  components?: string[];
  status: string;
}

export interface QuantDataStatus {
  status: string;
  validatedDatasets?: number;
  availableDatasets: number;
  backend: string;
  validationScope: string;
  checkedAt: string;
  datasets: QuantDataStatusDataset[];
  datasetSetVersion?: string;
  generation?: number;
  releaseSetSha256?: string;
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

function convertApiContract(data: unknown): QuantApiContract {
  const contract = toCamelCase<Partial<QuantApiContract>>(data);
  return {
    version: typeof contract.version === 'string' ? contract.version : '',
    backendBuildId: typeof contract.backendBuildId === 'string' ? contract.backendBuildId : '',
    backendStartedAt: typeof contract.backendStartedAt === 'string' ? contract.backendStartedAt : '',
  };
}

function convertRecordValues<T>(data: unknown, converter: (value: unknown) => T): Record<string, T> {
  if (!data || typeof data !== 'object') {
    return {};
  }
  return Object.fromEntries(
    Object.entries(data as Record<string, unknown>).map(([key, value]) => [key, converter(value)]),
  );
}

function convertLocalService(data: Record<string, unknown>): LocalServiceStatus {
  const converted = toCamelCase<Partial<LocalServiceStatus>>(data);
  const service = converted.service === 'catdesk' || converted.service === 'mlflow'
    ? converted.service
    : 'mlflow';
  return {
    service,
    status: typeof converted.status === 'string' ? converted.status : 'unavailable',
    url: typeof converted.url === 'string' ? converted.url : '',
    openUrl: typeof converted.openUrl === 'string' ? converted.openUrl : '',
    error: typeof converted.error === 'string' ? converted.error : null,
    pid: typeof converted.pid === 'number' ? converted.pid : null,
    port: typeof converted.port === 'number' ? converted.port : null,
    message: typeof converted.message === 'string' ? converted.message : null,
  };
}

function convertStatus(data: Record<string, unknown>): QuantStatus {
  const camel = toCamelCase<Partial<QuantStatus>>(data);
  return {
    apiContract: data.api_contract ? convertApiContract(data.api_contract) : undefined,
    availableJobTypes: convertRecordValues(data.available_job_types, convertJobTypeInfo),
    activeCount: Number(data.active_count || 0),
    latestJobs: convertRecordValues(data.latest_jobs, convertJob),
    recentJobs: Array.isArray(data.recent_jobs) ? data.recent_jobs.map(convertJob) : [],
    generatedAt: camel.generatedAt,
    trainable: Boolean(camel.trainable),
    trainingReadiness: camel.trainingReadiness,
    trainingPreflight: camel.trainingPreflight,
    trainingPreflights: camel.trainingPreflights ?? {},
    dataSources: Array.isArray(camel.dataSources) ? camel.dataSources : [],
    qlibProvider: camel.qlibProvider ?? null,
    providerQuality: camel.providerQuality ?? {},
    currentProviderIdentities: camel.currentProviderIdentities ?? {},
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
    if (!Array.isArray(payload.jobs)) {
      console.warn('[quant job stream] ignored event without a jobs array');
      return [];
    }
    return payload.jobs.map(convertJob);
  } catch {
    console.warn('[quant job stream] ignored event with invalid JSON');
    return [];
  }
}

export const quantApi = {
  async getDataStatus(): Promise<QuantDataStatus> {
    const response = await apiClient.get<Record<string, unknown>>('/api/v1/data/status');
    const converted = toCamelCase<Partial<QuantDataStatus>>(response.data);
    return {
      status: typeof converted.status === 'string' ? converted.status : 'unknown',
      availableDatasets: Number(converted.availableDatasets || 0),
      backend: typeof converted.backend === "string" ? converted.backend : "unknown",
      validationScope: typeof converted.validationScope === "string" ? converted.validationScope : "unknown",
      checkedAt: typeof converted.checkedAt === 'string' ? converted.checkedAt : '',
      datasets: Array.isArray(converted.datasets) ? converted.datasets : [],
    };
  },

  async getJobTypes(): Promise<QuantJobTypesCatalog> {
    const response = await apiClient.get<Record<string, unknown>>('/api/v1/quant/job-types');
    return {
      availableJobTypes: convertRecordValues(response.data.available_job_types, convertJobTypeInfo),
      apiContract: convertApiContract(response.data.api_contract),
    };
  },

  async getStatus(options: QuantStatusOptions = {}): Promise<QuantStatus> {
    const requestConfig = options.includeResearch === false
      ? { params: { include_research: false } }
      : undefined;
    const response = await apiClient.get<Record<string, unknown>>(
      '/api/v1/quant/status',
      requestConfig,
    );
    return convertStatus(response.data);
  },

  async getLocalServiceStatus(service: LocalServiceName): Promise<LocalServiceStatus> {
    const response = await apiClient.get<Record<string, unknown>>(
      `/api/v1/quant/services/${encodeURIComponent(service)}`,
    );
    return convertLocalService(response.data);
  },

  async startLocalService(service: LocalServiceName): Promise<LocalServiceStatus> {
    const response = await apiClient.post<Record<string, unknown>>(
      `/api/v1/quant/services/${encodeURIComponent(service)}/start`,
    );
    return convertLocalService(response.data);
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
