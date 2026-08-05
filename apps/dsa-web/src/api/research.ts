import apiClient from './index';
import { toCamelCase } from './utils';

export type ResearchSourceStatus = 'available' | 'partial' | 'missing' | string;

export type ResearchDataSource = {
  name: string;
  datasetId?: string;
  role?: 'canonical' | 'audit' | 'legacy' | string;
  lifecycle?: string;
  files: number;
  rows: number;
  dateRange: string;
  status: ResearchSourceStatus;
  sizeMb?: number;
};

export type ResearchTrainingReadiness = {
  status: string;
  summary: string;
  failureReasons?: string[];
};

export type ResearchProviderQuality = {
  provider?: string;
  datasetId?: string;
  snapshotId?: string;
  manifestSha256?: string;
  contentSha256?: string;
  calendarEnd?: string;
  trainable?: boolean;
  structuralStatus?: 'accepted' | 'blocked' | string;
  freshness?: {
    status: 'current' | 'stale' | 'future' | 'unknown' | string;
    calendarEnd?: string | null;
    asOfDate?: string;
    calendarAgeDays?: number | null;
    blocksTraining?: boolean;
  };
  trainingReadiness?: ResearchTrainingReadiness;
};

export type ResearchQualityScan = {
  mode: 'full' | 'sampled' | string;
  coverage: 'full' | 'partial' | string;
  verificationStatus: 'verified' | 'unverified' | string;
  requestedSample?: number;
  scannedInstruments?: number | null;
};

export type ResearchQlibProvider = {
  path?: string;
  calendarDays?: number;
  calendarRange?: string;
  instruments?: number;
  featureDirs?: number;
  sizeMb?: number;
  datasetId?: string;
  role?: string;
  status?: string;
  priceBasis?: string;
  providerSource?: string;
  latestTrainingProviderCurrent?: boolean;
  providerUriWarning?: string;
  providerError?: string;
  snapshotId?: string;
  manifestSha256?: string;
  contentSha256?: string;
};

export type ResearchProviderIdentity = {
  providerUri?: string;
  providerContract?: string;
  datasetId?: string;
  snapshotId?: string;
  contentSha256?: string;
  manifestSha256?: string;
};

export type ResearchTrainingResult = {
  status: string;
  experimentName?: string;
  experimentId?: string;
  runId?: string;
  provider?: string;
  providerUri?: string;
  qlibProvider?: string;
  providerStatus?: string;
  providerCurrent?: boolean;
  providerWarning?: string;
  providerIdentity?: ResearchProviderIdentity;
  sourceConfig?: string;
  runtimeConfig?: string;
  artifactSha256?: Record<string, string>;
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

export type ResearchCandidateMetrics = {
  annualReturn?: number | null;
  benchmarkAnnualReturn?: number | null;
  annualExcessReturn?: number | null;
  informationRatio?: number | null;
  sharpe?: number | null;
  maxDrawdown?: number | null;
  avgTurnover?: number | null;
  rankIc?: number | null;
};

export type ResearchCandidateModel = {
  id: string;
  label?: string;
  status: string;
  experimentId?: string | null;
  runId?: string | null;
  artifactSha256?: Record<string, string>;
  selection?: ResearchCandidateMetrics;
  holdout?: ResearchCandidateMetrics;
  error?: string;
};

export type ResearchCandidateStrategy = {
  modelId: string;
  strategyId: string;
  strategyLabel?: string;
  strategyConfig?: Record<string, unknown>;
  selection?: ResearchCandidateMetrics;
  holdout?: ResearchCandidateMetrics;
};

export type ResearchCandidatePipeline = {
  status: string;
  dryRun?: boolean;
  batchId?: string;
  experimentName?: string;
  provider?: string;
  providerStatus?: string;
  providerCurrent?: boolean;
  providerWarning?: string;
  providerIdentity?: ResearchProviderIdentity;
  providerLatestDay?: string;
  latestPredictionDay?: string;
  latestLabelDay?: string;
  periods?: Record<string, string[]>;
  selectedModels?: string[];
  models?: ResearchCandidateModel[];
  strategyEvaluations?: ResearchCandidateStrategy[];
  selectedCombination?: ResearchCandidateStrategy | null;
  partialDevelopmentLeader?: ResearchCandidateStrategy | null;
  holdoutVerdict?: { status?: string; reason?: string; note?: string };
  artifacts?: Record<string, string>;
};

export type ResearchRobustStage = {
  name: string;
  status: string;
  startedAt?: string;
  finishedAt?: string;
  returnCode?: number;
  reasonCode?: string;
};

export type ResearchRobustMetricMap = Record<string, number | string | boolean | null | undefined>;

export type ResearchRobustFactorQuality = ResearchRobustMetricMap & {
  processed?: ResearchRobustMetricMap;
};

export type ResearchRobustModelReport = {
  modelId?: string;
  eligible?: boolean;
  developmentScore?: number;
  stability?: {
    rankCorrelationMedian?: number;
    topkOverlapMean?: number;
  };
};

export type ResearchRobustStrategyEvaluation = {
  strategyId?: string;
  eligibleForFinalSelection?: boolean;
  acceptance?: {
    passed?: boolean;
    failedGates?: string[];
    values?: ResearchRobustMetricMap;
    thresholds?: ResearchRobustMetricMap;
  };
  baseline?: { metrics?: ResearchRobustMetricMap };
  stress?: { metrics?: ResearchRobustMetricMap };
  delayed?: { metrics?: ResearchRobustMetricMap };
  quarterMetrics?: ResearchRobustMetricMap[];
  regimeMetrics?: Record<string, ResearchRobustMetricMap>;
  boardAttribution?: Record<string, ResearchRobustMetricMap>;
  attributionReconciliation?: ResearchRobustMetricMap;
};

export type ResearchRobustStrategy = {
  status: string;
  lifecycleStatus?: string;
  reasonCode?: string;
  runId?: string;
  currentStage?: string;
  failedStage?: string;
  updatedAt?: string;
  progress?: {
    stage: string;
    status?: string;
    completedDateCount: number;
    expectedDateCount: number;
    generatedAt?: string;
  } | null;
  stages: ResearchRobustStage[];
  preflight?: {
    status?: string;
    reasonCode?: string;
    generatedAt?: string;
    asOfDate?: string;
    summary?: Record<string, unknown>;
    checks?: Record<string, unknown> | unknown[];
  } | null;
  development?: {
    status?: string;
    reasonCode?: string;
    generatedAt?: string;
    asOfDate?: string;
    runId?: string;
    trainingJobCount?: number;
    selectedModels?: string[];
    selectedStrategy?: string;
    modelStability?: ResearchRobustModelReport[];
    strategyMatrix?: ResearchRobustStrategyEvaluation[];
  } | null;
  confirmation?: {
    status?: string;
    reasonCode?: string;
    generatedAt?: string;
    runId?: string;
    confirmationStart?: string;
    latestSignalDay?: string;
    latestMarketDay?: string;
    selectedModels?: string[];
    selectedStrategy?: string;
    metrics?: ResearchRobustMetricMap;
    verdict?: { passed?: boolean; failedGates?: string[] };
  } | null;
  shadow?: {
    status?: string;
    confirmationRunId?: string;
    startAfter?: string;
    createdAt?: string;
    updatedAt?: string;
    metrics?: ResearchRobustMetricMap;
    verdict?: { passed?: boolean; remainingDays?: number; failedGates?: string[] };
  } | null;
  shadowDaily?: {
    status?: string;
    reasonCode?: string;
    asOf?: string;
    outcome?: string;
    inferenceSuccess?: boolean;
    appendedDays?: number;
    generatedAt?: string;
    stateStatus?: string;
    stateMetrics?: ResearchRobustMetricMap;
    errorType?: string;
  } | null;
  mlflowExperiments?: string[];
};

export type ResearchRobustSignalReport = {
  phase?: string;
  signalId?: string;
  modelId?: string;
  passed?: boolean;
  failedGates?: string[];
  values?: ResearchRobustMetricMap;
  quarterRankIc?: Record<string, number | null>;
  windowStability?: ResearchRobustMetricMap;
  topkSensitivity?: {
    primaryTopk?: number;
    results?: Record<string, ResearchRobustMetricMap>;
  };
  predictionDispersion?: ResearchRobustMetricMap;
  monotonicity?: ResearchRobustMetricMap;
  returnsAfter20bps?: ResearchRobustMetricMap;
  boardAttribution?: Record<string, ResearchRobustMetricMap>;
  marketStateAttribution?: Record<string, ResearchRobustMetricMap>;
  factorQuality?: ResearchRobustFactorQuality;
};

export type ResearchRobustSignal = {
  status: string;
  lifecycleStatus?: string;
  reasonCode?: string;
  runId?: string;
  currentStage?: string;
  pauseState?: string;
  nextGate?: string;
  completedCount?: number;
  expectedCount?: number;
  updatedAt?: string;
  configHash?: string;
  gitCommit?: string;
  dataManifestHash?: string;
  contextHash?: string;
  dataReadiness?: {
    contractVersion?: string;
    status: string;
    trainable: boolean;
    reasonCode?: string;
    summary?: string;
    summaryZh?: string;
    generatedAt?: string;
    providerName?: string;
    providerKey?: string;
    providerTreeSha256?: string;
    dataManifestSha256?: string;
    membershipIntervalsSha256?: string;
    canonicalSnapshotId?: string;
    symbols?: number;
    intervals?: number;
  };
  progress?: {
    currentStage?: string;
    pauseState?: string;
    completedCount?: number;
    expectedCount?: number;
    currentJob?: string;
  };
  smokeRun?: {
    status: string;
    lifecycleStatus?: string;
    reasonCode?: string;
    runId?: string;
    currentStage?: string;
    pauseState?: string;
    nextGate?: string;
    completedCount?: number;
    expectedCount?: number;
    updatedAt?: string;
    configHash?: string;
    gitCommit?: string;
    dataManifestHash?: string;
    contextHash?: string;
    testOnly?: boolean;
    progress?: {
      currentStage?: string;
      pauseState?: string;
      completedCount?: number;
      expectedCount?: number;
      currentJob?: string;
    };
  };
  result?: {
    status?: string;
    reasonCode?: string;
    screeningCandidates?: string[];
    selectedSignalId?: string;
    selectedModelId?: string;
    challengers?: string;
  };
  factorContract?: {
    status?: string;
    signalId?: string;
    modelId?: string;
    featureFamily?: string;
    featurePrefixes?: string[];
    horizonDays?: number;
    labelExpression?: string;
    signalCutoff?: string;
    executionStart?: string;
    normalizationScope?: string;
    target?: string;
    contractHash?: string;
    offlineDailyParityRequired?: boolean;
  };
  screeningReports?: ResearchRobustSignalReport[];
  validationReports?: ResearchRobustSignalReport[];
};

export type ResearchQlibResearchSection = {
  section: string;
  title: string;
  status: string;
  generatedAt?: string;
  mode?: string;
  path?: string | null;
  artifacts?: Record<string, string>;
  run?: Record<string, unknown>;
  metrics?: Record<string, unknown>;
  scoreFactor?: Record<string, unknown>;
  riskMetrics?: Record<string, unknown>;
  nextActions?: string[];
  limitations?: string[];
};

export type ResearchQlibResearch = {
  status: string;
  availableSections: number;
  totalSections: number;
  sections: Record<string, ResearchQlibResearchSection>;
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

export type ResearchMlflow = {
  url: string;
  status?: string;
  role?: string;
  checkedAt?: string | null;
  healthChecked?: boolean;
  error?: string | null;
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

export type ResearchDataHygieneInventory = {
  status: string;
  validationStatus?: string;
  path?: string | null;
  generatedAt?: string | null;
  officialEntries?: number;
  candidateEntries?: number;
  legacyEntries?: number;
  archiveCandidates?: number;
};

export type ResearchDataHygieneCleanupAction = {
  action: string;
  source: string;
  destination?: string;
  reason: string;
  risk?: string;
  requiresConfirmation: boolean;
};

export type ResearchDataHygieneCleanupPlan = {
  status: string;
  path?: string | null;
  generatedAt?: string | null;
  dryRunOnly?: boolean;
  totalActions?: number;
  previewLimit?: number;
  truncated?: boolean;
  actions?: ResearchDataHygieneCleanupAction[];
};

export type ResearchDataHygieneQualitySummary = {
  stockUniverse?: string;
  usableInstruments?: string;
  totalRows?: string;
  dateRange?: string;
};

export type ResearchDataHygieneReport = {
  status: string;
  path?: string | null;
  summary?: ResearchDataHygieneQualitySummary | null;
};

export type ResearchDataHygieneManifest = {
  status: string;
  validationStatus?: string;
  path?: string | null;
  generatedAt?: string | null;
  sourceReport?: string | null;
  officialEntries?: number;
};

export type ResearchDataHygieneMaintenanceLock = {
  locked: boolean;
  path: string;
  updatedAt?: string | null;
};

export type ResearchDataHygiene = {
  inventory: ResearchDataHygieneInventory;
  cleanupPlan: ResearchDataHygieneCleanupPlan;
  qualityReport: ResearchDataHygieneReport;
  cleaningStrategy: ResearchDataHygieneReport;
  manifest?: ResearchDataHygieneManifest | null;
  readme?: ResearchDataHygieneReport | null;
  maintenanceLock: ResearchDataHygieneMaintenanceLock;
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
  qualityScan?: ResearchQualityScan;
  dataSources: ResearchDataSource[];
  qlibProvider?: ResearchQlibProvider | null;
  providerQuality?: Record<string, ResearchProviderQuality>;
  latestTrainingProvider?: ResearchQlibProvider | null;
  latestTraining?: ResearchTrainingResult | null;
  cashControlBacktest?: ResearchCashControlBacktest | null;
  qlibPredictions?: ResearchQlibPredictions | null;
  candidatePipeline?: ResearchCandidatePipeline | null;
  robustStrategy?: ResearchRobustStrategy | null;
  robustSignal?: ResearchRobustSignal | null;
  qlibResearch?: ResearchQlibResearch | null;
  experimentNotebook?: ResearchExperimentNotebook | null;
  mlflow?: ResearchMlflow | null;
  dataHygiene?: ResearchDataHygiene | null;
  externalTools?: ResearchExternalTools | null;
  capabilityMatrix?: ResearchCapabilityMatrixItem[] | null;
  functionInventory?: ResearchFunctionInventoryItem[] | null;
  chinaAstockReuse?: ResearchChinaAstockReuseItem[] | null;
  acceptanceChecklist?: ResearchAcceptanceChecklistItem[] | null;
};

export function convertQlibResearch(data: unknown): ResearchQlibResearch | null {
  if (!data || typeof data !== 'object') {
    return null;
  }
  const raw = data as Record<string, unknown>;
  const camel = toCamelCase<Omit<ResearchQlibResearch, 'sections'> & { sections?: unknown }>(raw);
  const rawSections = raw.sections;
  const sections = rawSections && typeof rawSections === 'object'
    ? Object.fromEntries(
        Object.entries(rawSections as Record<string, unknown>).map(([key, value]) => [
          key,
          toCamelCase<ResearchQlibResearchSection>(value),
        ]),
      )
    : {};

  return {
    status: camel.status || 'missing',
    availableSections: Number(camel.availableSections || 0),
    totalSections: Number(camel.totalSections || Object.keys(sections).length),
    sections,
  };
}

function convertResearchStatus(data: Record<string, unknown>): ResearchStatus {
  const status = toCamelCase<ResearchStatus>(data);
  status.qlibResearch = convertQlibResearch(data.qlib_research);
  const rawLatestTraining = data.latest_training;
  if (
    status.latestTraining
    && rawLatestTraining
    && typeof rawLatestTraining === 'object'
    && !Array.isArray(rawLatestTraining)
  ) {
    const artifactSha256 = preserveArtifactSha256(
      (rawLatestTraining as Record<string, unknown>).artifact_sha256,
    );
    if (artifactSha256) status.latestTraining.artifactSha256 = artifactSha256;
  }

  const rawCandidatePipeline = data.candidate_pipeline;
  if (
    status.candidatePipeline?.models
    && rawCandidatePipeline
    && typeof rawCandidatePipeline === 'object'
    && !Array.isArray(rawCandidatePipeline)
  ) {
    const rawModels = (rawCandidatePipeline as Record<string, unknown>).models;
    if (Array.isArray(rawModels)) {
      for (const rawModel of rawModels) {
        if (!rawModel || typeof rawModel !== 'object' || Array.isArray(rawModel)) continue;
        const raw = rawModel as Record<string, unknown>;
        const model = status.candidatePipeline.models.find((item) => item.id === raw.id);
        const artifactSha256 = preserveArtifactSha256(raw.artifact_sha256);
        if (model && artifactSha256) model.artifactSha256 = artifactSha256;
      }
    }
  }
  return status;
}

function preserveArtifactSha256(value: unknown): Record<string, string> | undefined {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return undefined;
  const entries = Object.entries(value).filter(
    (entry): entry is [string, string] => typeof entry[1] === 'string',
  );
  return entries.length > 0 ? Object.fromEntries(entries) : undefined;
}

export const researchApi = {
  async getStatus(): Promise<ResearchStatus> {
    const response = await apiClient.get<Record<string, unknown>>('/api/v1/research/status');
    return convertResearchStatus(response.data);
  },

  async getExperimentNotebook(): Promise<ResearchExperimentNotebook> {
    const response = await apiClient.get<Record<string, unknown>>('/api/v1/research/experiments');
    return toCamelCase<ResearchExperimentNotebook>(response.data);
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
      { params: { confirm: 'permanent' } },
    );
    return toCamelCase<ResearchExperimentMutationResult>(response.data);
  },
};
