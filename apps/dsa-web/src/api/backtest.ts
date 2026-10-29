import apiClient from './index';
import { toCamelCase } from './utils';
import type {
  BacktestRunRequest,
  BacktestRunResponse,
  BacktestResultsResponse,
  BacktestResultItem,
  PerformanceMetrics,
  BacktestPhaseFilter,
  StabilityTestRequest,
  StabilityTestResponse,
  StrategyBacktestResponse,
  StrategyBacktestRunRequest,
  FormStrategyBacktestRunRequest,
  ScriptStrategyBacktestRunRequest,
  TrainedStrategyBacktestResponse,
  TrainedStrategyBacktestRunRequest,
  TrainedStrategyCatalogResponse,
  StrategyAssetSummary,
  StrategyAssetDetail,
  StrategyAssetCreateRequest,
  StrategyAssetUpdateRequest,
  BacktestRunSummary,
  BacktestRunDetail,
} from '../types/backtest';

// ============ API ============

const TRAINED_STRATEGY_REPLAY_TIMEOUT_MS = 120_000;

export const backtestApi = {
  getTrainedStrategyCatalog: async (): Promise<TrainedStrategyCatalogResponse> => {
    const response = await apiClient.get<Record<string, unknown>>(
      '/api/v1/backtest/strategy/trained/catalog',
    );
    return toCamelCase<TrainedStrategyCatalogResponse>(response.data);
  },

  runTrainedStrategy: async (
    params: TrainedStrategyBacktestRunRequest,
  ): Promise<TrainedStrategyBacktestResponse> => {
    const requestData: Record<string, unknown> = {
      run_id: params.runId,
      model_id: params.modelId,
      strategy_id: params.strategyId,
      replay_mode: params.replayMode,
    };
    if (params.replayMode === 'custom') {
      requestData.start_date = params.startDate;
      requestData.end_date = params.endDate;
      requestData.topk = params.topk;
      requestData.cost_bps = params.costBps;
      if (params.initialCapital !== undefined) requestData.initial_capital = params.initialCapital;
      if (params.lotSize !== undefined) requestData.lot_size = params.lotSize;
    }
    const response = await apiClient.post<Record<string, unknown>>(
      '/api/v1/backtest/strategy/trained',
      requestData,
      { timeout: TRAINED_STRATEGY_REPLAY_TIMEOUT_MS },
    );
    return toCamelCase<TrainedStrategyBacktestResponse>(response.data);
  },

  runFormStrategy: async (
    params: FormStrategyBacktestRunRequest,
  ): Promise<StrategyBacktestResponse> => {
    const response = await apiClient.post<Record<string, unknown>>(
      '/api/v1/backtest/strategy/run',
      {
        strategy_id: params.strategyId,
        symbols: params.symbols,
        start_date: params.startDate || undefined,
        end_date: params.endDate || undefined,
        initial_capital: params.initialCapital,
        side_cost_pct: params.sideCostPct,
        lot_size: params.lotSize,
        lookback_days: params.lookbackDays,
        moving_average_days: params.movingAverageDays,
        fast_ma_days: params.fastMaDays,
        slow_ma_days: params.slowMaDays,
        top_n: params.topN,
      },
    );
    return toCamelCase<StrategyBacktestResponse>(response.data);
  },

  runScriptStrategy: async (
    params: ScriptStrategyBacktestRunRequest,
  ): Promise<StrategyBacktestResponse> => {
    const response = await apiClient.post<Record<string, unknown>>(
      '/api/v1/backtest/strategy/script',
      {
        script: params.script,
        pool: params.pool || 'etf',
        symbols: params.symbols || [],
        benchmarks: params.benchmarks || [],
        comparison_strategy_ids: params.comparisonStrategyIds || [],
        comparison_presets: params.comparisonPresets || [],
        start_date: params.startDate || undefined,
        end_date: params.endDate || undefined,
        initial_capital: params.initialCapital,
        side_cost_pct: params.sideCostPct,
        lot_size: params.lotSize,
        strategy_asset_id: params.strategyAssetId || undefined,
      },
      { timeout: 600_000 },
    );
    return toCamelCase<StrategyBacktestResponse>(response.data);
  },

  runStrategy: async (
    params: StrategyBacktestRunRequest,
  ): Promise<StrategyBacktestResponse> => {
    const requestData: Record<string, unknown> = {
      code: params.code,
      initial_capital: params.initialCapital,
      side_cost_pct: params.sideCostPct,
      lot_size: params.lotSize,
    };
    if (params.startDate) requestData.start_date = params.startDate;
    if (params.endDate) requestData.end_date = params.endDate;

    const response = await apiClient.post<Record<string, unknown>>(
      '/api/v1/backtest/strategy/dsa-signal',
      requestData,
    );
    return toCamelCase<StrategyBacktestResponse>(response.data);
  },

  /**
   * Trigger backtest evaluation
   */
  run: async (params: BacktestRunRequest = {}): Promise<BacktestRunResponse> => {
    const requestData: Record<string, unknown> = {};
    if (params.code?.trim()) requestData.code = params.code.trim();
    if (params.force) requestData.force = params.force;
    if (params.evalWindowDays != null) requestData.eval_window_days = params.evalWindowDays;
    if (params.minAgeDays != null) requestData.min_age_days = params.minAgeDays;
    if (params.analysisDateFrom) requestData.analysis_date_from = params.analysisDateFrom;
    if (params.analysisDateTo) requestData.analysis_date_to = params.analysisDateTo;
    if (params.limit != null) requestData.limit = params.limit;

    const response = await apiClient.post<Record<string, unknown>>(
      '/api/v1/backtest/run',
      requestData,
    );
    return toCamelCase<BacktestRunResponse>(response.data);
  },

  /**
   * Get paginated backtest results
   */
  getResults: async (params: {
    code?: string;
    evalWindowDays?: number;
    analysisDateFrom?: string;
    analysisDateTo?: string;
    analysisPhase?: BacktestPhaseFilter;
    page?: number;
    limit?: number;
  } = {}): Promise<BacktestResultsResponse> => {
    const { code, evalWindowDays, analysisDateFrom, analysisDateTo, analysisPhase, page = 1, limit = 20 } = params;

    const queryParams: Record<string, string | number> = { page, limit };
    if (code) queryParams.code = code;
    if (evalWindowDays) queryParams.eval_window_days = evalWindowDays;
    if (analysisDateFrom) queryParams.analysis_date_from = analysisDateFrom;
    if (analysisDateTo) queryParams.analysis_date_to = analysisDateTo;
    if (analysisPhase && analysisPhase !== 'all') queryParams.analysis_phase = analysisPhase;

    const response = await apiClient.get<Record<string, unknown>>(
      '/api/v1/backtest/results',
      { params: queryParams },
    );

    const data = toCamelCase<BacktestResultsResponse>(response.data);
    return {
      total: data.total,
      page: data.page,
      limit: data.limit,
      items: (data.items || []).map(item => toCamelCase<BacktestResultItem>(item)),
    };
  },

  /**
   * Get overall performance metrics
   */
  getOverallPerformance: async (params: {
    evalWindowDays?: number;
    analysisDateFrom?: string;
    analysisDateTo?: string;
    analysisPhase?: BacktestPhaseFilter;
  } = {}): Promise<PerformanceMetrics | null> => {
    try {
      const queryParams: Record<string, string | number | boolean> = { empty_ok: true };
      if (params.evalWindowDays) queryParams.eval_window_days = params.evalWindowDays;
      if (params.analysisDateFrom) queryParams.analysis_date_from = params.analysisDateFrom;
      if (params.analysisDateTo) queryParams.analysis_date_to = params.analysisDateTo;
      if (params.analysisPhase && params.analysisPhase !== 'all') queryParams.analysis_phase = params.analysisPhase;
      const response = await apiClient.get<Record<string, unknown> | null>(
        '/api/v1/backtest/performance',
        { params: queryParams },
      );
      if (response.data == null) return null;
      return toCamelCase<PerformanceMetrics>(response.data);
    } catch (err: unknown) {
      if (err && typeof err === 'object' && 'response' in err) {
        const axiosErr = err as { response?: { status?: number } };
        if (axiosErr.response?.status === 404) return null;
      }
      throw err;
    }
  },

  /**
   * Get per-stock performance metrics
   */
  getStockPerformance: async (code: string, params: {
    evalWindowDays?: number;
    analysisDateFrom?: string;
    analysisDateTo?: string;
    analysisPhase?: BacktestPhaseFilter;
  } = {}): Promise<PerformanceMetrics | null> => {
    try {
      const queryParams: Record<string, string | number | boolean> = { empty_ok: true };
      if (params.evalWindowDays) queryParams.eval_window_days = params.evalWindowDays;
      if (params.analysisDateFrom) queryParams.analysis_date_from = params.analysisDateFrom;
      if (params.analysisDateTo) queryParams.analysis_date_to = params.analysisDateTo;
      if (params.analysisPhase && params.analysisPhase !== 'all') queryParams.analysis_phase = params.analysisPhase;
      const response = await apiClient.get<Record<string, unknown> | null>(
        `/api/v1/backtest/performance/${encodeURIComponent(code)}`,
        { params: queryParams },
      );
      if (response.data == null) return null;
      return toCamelCase<PerformanceMetrics>(response.data);
    } catch (err: unknown) {
      if (err && typeof err === 'object' && 'response' in err) {
        const axiosErr = err as { response?: { status?: number } };
        if (axiosErr.response?.status === 404) return null;
      }
      throw err;
    }
  },

  listStrategyAssets: async (params: { archived?: boolean } = {}): Promise<StrategyAssetSummary[]> => {
    const response = await apiClient.get<Record<string, unknown>[]>(
      '/api/v1/backtest/strategies',
      { params: { archived: params.archived ?? false } },
    );
    return response.data.map((item) => toCamelCase<StrategyAssetSummary>(item));
  },

  getStrategyAsset: async (strategyId: string): Promise<StrategyAssetDetail> => {
    const response = await apiClient.get<Record<string, unknown>>(
      `/api/v1/backtest/strategies/${encodeURIComponent(strategyId)}`,
    );
    return toCamelCase<StrategyAssetDetail>(response.data);
  },

  saveStrategyAsset: async (
    request: StrategyAssetCreateRequest,
  ): Promise<StrategyAssetDetail> => {
    const response = await apiClient.post<Record<string, unknown>>(
      '/api/v1/backtest/strategies',
      {
        name: request.name,
        description: request.description || '',
        script: request.script,
      },
    );
    return toCamelCase<StrategyAssetDetail>(response.data);
  },

  updateStrategyAsset: async (
    strategyId: string,
    request: StrategyAssetUpdateRequest,
  ): Promise<StrategyAssetDetail> => {
    const response = await apiClient.patch<Record<string, unknown>>(
      `/api/v1/backtest/strategies/${encodeURIComponent(strategyId)}`,
      {
        name: request.name,
        description: request.description,
        script: request.script,
        archived: request.archived,
      },
    );
    return toCamelCase<StrategyAssetDetail>(response.data);
  },

  deleteStrategyAsset: async (strategyId: string): Promise<void> => {
    await apiClient.delete(`/api/v1/backtest/strategies/${encodeURIComponent(strategyId)}`);
  },

  listBacktestRuns: async (
    params: {
      limit?: number;
      offset?: number;
      strategyId?: string;
      scriptSha256?: string;
    } = {},
  ): Promise<BacktestRunSummary[]> => {
    const queryParams: Record<string, string | number> = {};
    if (params.limit) queryParams.limit = params.limit;
    if (params.offset) queryParams.offset = params.offset;
    if (params.strategyId) queryParams.strategy_id = params.strategyId;
    if (params.scriptSha256) queryParams.script_sha256 = params.scriptSha256;
    const response = await apiClient.get<Record<string, unknown>[]>(
      '/api/v1/backtest/runs',
      { params: queryParams },
    );
    return response.data.map((item) => toCamelCase<BacktestRunSummary>(item));
  },

  getBacktestRun: async (runId: string): Promise<BacktestRunDetail> => {
    const response = await apiClient.get<Record<string, unknown>>(
      `/api/v1/backtest/runs/${encodeURIComponent(runId)}`,
    );
    return toCamelCase<BacktestRunDetail>(response.data);
  },

  deleteBacktestRun: async (runId: string): Promise<void> => {
    await apiClient.delete(`/api/v1/backtest/runs/${encodeURIComponent(runId)}`);
  },

  runStabilityTest: async (
    params: StabilityTestRequest,
  ): Promise<StabilityTestResponse> => {
    const response = await apiClient.post<Record<string, unknown>>(
      '/api/v1/backtest/strategy/script/stability',
      {
        script: params.script,
        pool: params.pool || 'etf',
        symbols: params.symbols || [],
        start_date: params.startDate || undefined,
        end_date: params.endDate || undefined,
        initial_capital: params.initialCapital,
        side_cost_pct: params.sideCostPct,
        lot_size: params.lotSize,
        min_window_days: params.minWindowDays ?? 252,
        iterations: params.iterations ?? 20,
        seed: params.seed ?? null,
      },
      { timeout: 600_000 },
    );
    return toCamelCase<StabilityTestResponse>(response.data);
  },
};
