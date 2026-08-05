import type React from 'react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  FlaskConical,
  Loader2,
  Play,
  RotateCcw,
  Upload,
} from 'lucide-react';
import { useSearchParams } from 'react-router-dom';
import { backtestApi } from '../../api/backtest';
import { getParsedApiError } from '../../api/error';
import {
  researchApi,
  type ResearchCandidateMetrics,
  type ResearchCandidateStrategy,
  type ResearchStatus,
} from '../../api/research';
import type {
  TrainedStrategyBacktestResponse,
  TrainedStrategyCatalogResponse,
  TrainedStrategyCatalogRun,
  TrainedStrategyCatalogStrategy,
} from '../../types/backtest';
import { providerStatusNotice } from '../../utils/providerStatus';
import { Badge, InlineAlert } from '../common';
import TrainedStrategyBacktestResults from './TrainedStrategyBacktestResults';

type ModelStrategyBacktestPanelProps = {
  language: 'zh' | 'en';
};

type LoadedSelection = {
  runKey: string;
  strategyId: string;
};

type RequestedSelection = {
  modelId: string | null;
  strategyId: string | null;
  runId: string | null;
};

function makeRunKey(run: Pick<TrainedStrategyCatalogRun, 'runId' | 'modelId'>): string {
  return `${run.runId}:${run.modelId}`;
}

function normalizeStrategyId(strategyId: string | null): string | null {
  return strategyId === 'qlib_default' ? 'baseline_daily' : strategyId;
}

function isBacktestSelectableRun(run: TrainedStrategyCatalogRun): boolean {
  return run.replayCapability !== 'unavailable'
    && run.strategies.length > 0
    && (run.customReplayAvailable || run.strategies.some((strategy) => strategy.exactReplayAvailable));
}

function findRequestedSelection(
  catalog: TrainedStrategyCatalogResponse,
  requested: RequestedSelection,
): LoadedSelection | null {
  const requestedStrategy = normalizeStrategyId(requested.strategyId);
  const hasRequestedIdentity = requested.runId !== null || requested.modelId !== null;
  if (hasRequestedIdentity) {
    const requestedRun = catalog.runs.find((run) => (
      isBacktestSelectableRun(run)
      && run.runId === requested.runId
      && run.modelId === requested.modelId
    ));
    if (!requestedRun) return null;

    const strategy = requestedStrategy
      ? requestedRun.strategies.find((item) => item.id === requestedStrategy)
      : requestedRun.strategies.find((item) => item.id === requestedRun.originalParameters.strategyId)
        ?? requestedRun.strategies[0];
    return strategy
      ? { runKey: makeRunKey(requestedRun), strategyId: strategy.id }
      : null;
  }

  const first = catalog.runs.find(isBacktestSelectableRun);
  if (!first) return null;
  const originalStrategy = first.originalParameters.strategyId;
  const strategy = first.strategies.find((item) => item.id === originalStrategy)
    ?? first.strategies[0];
  return { runKey: makeRunKey(first), strategyId: strategy.id };
}

function formatPercent(value?: number | null): string {
  return typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(2)}%` : '-';
}

function formatNumber(value?: number | null): string {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(2) : '-';
}

function strategyName(strategy: TrainedStrategyCatalogStrategy | undefined, language: 'zh' | 'en'): string {
  if (!strategy) return '-';
  if (language === 'en') {
    const known: Record<string, string> = {
      baseline_daily: 'Daily equal-weight TopK baseline',
      low_turnover_weekly: 'Weekly low turnover',
      ma200_weekly: 'MA200 weekly',
      volatility_weekly: 'Volatility-controlled weekly',
      combined_risk: 'Combined risk control',
    };
    return known[strategy.id] ?? strategy.label;
  }
  return strategy.label;
}

function runOptionLabel(run: TrainedStrategyCatalogRun, language: 'zh' | 'en'): string {
  const date = run.generatedAt?.slice(0, 10) || (language === 'zh' ? '日期未知' : 'date unknown');
  return `${run.modelLabel} · ${date} · Run ${run.runId}`;
}

const MetricCell: React.FC<{
  label: string;
  value: string;
  tone?: 'positive' | 'negative' | 'neutral';
}> = ({ label, value, tone = 'neutral' }) => (
  <div className="border-r border-border/60 px-3 py-3 last:border-r-0">
    <p className="text-xs text-muted-text">{label}</p>
    <p className={`mt-1 font-mono text-base font-semibold ${tone === 'positive' ? 'text-danger' : tone === 'negative' ? 'text-success' : 'text-foreground'}`}>
      {value}
    </p>
  </div>
);

const PeriodMetrics: React.FC<{
  title: string;
  metrics?: ResearchCandidateMetrics;
  language: 'zh' | 'en';
}> = ({ title, metrics, language }) => {
  const zh = language === 'zh';
  const excess = metrics?.annualExcessReturn;
  return (
    <section className="border border-border/70 bg-card/30">
      <h3 className="border-b border-border/60 px-3 py-2 text-sm font-medium text-foreground">{title}</h3>
      <div className="grid grid-cols-2 sm:grid-cols-3">
        <MetricCell
          label={zh ? `${title}年化超额` : `${title} annual excess`}
          value={formatPercent(excess)}
          tone={typeof excess === 'number' ? (excess >= 0 ? 'positive' : 'negative') : 'neutral'}
        />
        <MetricCell label={zh ? '年化收益' : 'Annual return'} value={formatPercent(metrics?.annualReturn)} />
        <MetricCell label={zh ? '基准年化收益' : 'Benchmark annual return'} value={formatPercent(metrics?.benchmarkAnnualReturn)} />
        <MetricCell label={zh ? '信息比率' : 'Information ratio'} value={formatNumber(metrics?.informationRatio)} />
        <MetricCell label={zh ? '夏普比率' : 'Sharpe ratio'} value={formatNumber(metrics?.sharpe)} />
        <MetricCell label={zh ? '最大回撤' : 'Maximum drawdown'} value={formatPercent(metrics?.maxDrawdown)} tone="negative" />
      </div>
    </section>
  );
};

const ModelStrategyBacktestPanel: React.FC<ModelStrategyBacktestPanelProps> = ({ language }) => {
  const zh = language === 'zh';
  const [searchParams, setSearchParams] = useSearchParams();
  const [status, setStatus] = useState<ResearchStatus | null>(null);
  const [catalog, setCatalog] = useState<TrainedStrategyCatalogResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectionError, setSelectionError] = useState(false);
  const [draftRunKey, setDraftRunKey] = useState('');
  const [draftStrategyId, setDraftStrategyId] = useState('');
  const [loaded, setLoaded] = useState<LoadedSelection | null>(null);
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [topk, setTopk] = useState('50');
  const [costBps, setCostBps] = useState('15');
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [runResult, setRunResult] = useState<TrainedStrategyBacktestResponse | null>(null);
  const runRequestVersionRef = useRef(0);
  const requestedModelId = searchParams.get('modelId');
  const requestedStrategyId = searchParams.get('strategyId');
  const requestedRunId = searchParams.get('runId');
  const selectableRuns = useMemo(
    () => (catalog?.runs ?? []).filter(isBacktestSelectableRun),
    [catalog],
  );
  const unavailableRuns = useMemo(
    () => (catalog?.runs ?? []).filter((run) => !isBacktestSelectableRun(run)),
    [catalog],
  );
  const unavailableReasons = useMemo(() => Array.from(new Set(
    unavailableRuns.map((run) => (
      zh
        ? run.replayUnavailableReason || '该 Run 缺少完整回测契约，不能回测。'
        : 'This Run lacks an immutable data identity or frozen trading-strategy contract and cannot be backtested.'
    )),
  )), [unavailableRuns, zh]);

  const applyOriginalParameters = useCallback((run: TrainedStrategyCatalogRun) => {
    setStartDate(run.originalParameters.startDate || '');
    setEndDate(run.originalParameters.endDate || '');
    setTopk(String(run.originalParameters.topk ?? 50));
    setCostBps(String(run.originalParameters.costBps ?? 15));
  }, []);

  useEffect(() => {
    let active = true;
    Promise.all([
      backtestApi.getTrainedStrategyCatalog(),
      researchApi.getStatus().catch(() => null),
    ])
      .then(([catalogPayload, statusPayload]) => {
        if (!active) return;
        setCatalog(catalogPayload);
        setStatus(statusPayload);
        setError(null);
      })
      .catch((reason: unknown) => {
        if (!active) return;
        const parsed = getParsedApiError(reason);
        setError(parsed.message || parsed.rawMessage || 'load_failed');
        setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!catalog) return;
    const next = findRequestedSelection(catalog, {
      modelId: requestedModelId,
      strategyId: requestedStrategyId,
      runId: requestedRunId,
    });
    if (!next) {
      runRequestVersionRef.current += 1;
      setRunning(false);
      setRunResult(null);
      setRunError(null);
      setDraftRunKey('');
      setDraftStrategyId('');
      setLoaded(null);
      setSelectionError(requestedModelId !== null || requestedRunId !== null);
      setLoading(false);
      return;
    }
    const run = catalog.runs.find((item) => makeRunKey(item) === next.runKey);
    runRequestVersionRef.current += 1;
    setRunning(false);
    setRunResult(null);
    setRunError(null);
    setDraftRunKey(next.runKey);
    setDraftStrategyId(next.strategyId);
    setLoaded(next);
    setSelectionError(false);
    if (run) applyOriginalParameters(run);
    setLoading(false);
  }, [applyOriginalParameters, catalog, requestedModelId, requestedRunId, requestedStrategyId]);

  const draftRun = selectableRuns.find((run) => makeRunKey(run) === draftRunKey);
  const loadedRun = selectableRuns.find((run) => makeRunKey(run) === loaded?.runKey);
  const draftStrategies = draftRun?.strategies ?? [];
  const loadedCatalogStrategy = loadedRun?.strategies.find((row) => row.id === loaded?.strategyId);
  const loadedExactReplayAvailable = loadedCatalogStrategy?.exactReplayAvailable ?? false;
  const currentCandidateModel = status?.candidatePipeline?.models?.find((model) => (
    model.id === loadedRun?.modelId && model.runId === loadedRun?.runId
  ));
  const currentCandidateStrategy: ResearchCandidateStrategy | undefined = currentCandidateModel
    ? status?.candidatePipeline?.strategyEvaluations?.find((row) => (
        row.modelId === loadedRun?.modelId && row.strategyId === loaded?.strategyId
      ))
    : undefined;
  const isCurrentBaseline = Boolean(
    loadedRun?.sourceType === 'baseline'
    && status?.latestTraining?.runId === loadedRun.runId,
  );
  const loadedArtifact = loadedRun?.sourceType === 'baseline' ? 'training' : 'candidate';
  const loadedProvider = isCurrentBaseline
    ? status?.latestTraining
    : currentCandidateModel
      ? status?.candidatePipeline
      : undefined;
  const loadedProviderNotice = loadedRun && loadedProvider
    ? providerStatusNotice(loadedProvider, loadedArtifact, language)
    : null;

  const handleRunChange = (nextKey: string) => {
    setDraftRunKey(nextKey);
    const run = selectableRuns.find((item) => makeRunKey(item) === nextKey);
    const preferred = run?.strategies.find((item) => item.id === run.originalParameters.strategyId)
      ?? run?.strategies[0];
    setDraftStrategyId(preferred?.id || '');
  };

  const handleLoad = () => {
    if (!draftRun || !draftStrategyId) return;
    const next = { runKey: makeRunKey(draftRun), strategyId: draftStrategyId };
    runRequestVersionRef.current += 1;
    setRunning(false);
    setLoaded(next);
    applyOriginalParameters(draftRun);
    setRunResult(null);
    setRunError(null);
    const nextParams = new URLSearchParams(searchParams);
    nextParams.set('tab', 'trained');
    nextParams.set('runId', draftRun.runId);
    nextParams.set('modelId', draftRun.modelId);
    nextParams.set('strategyId', draftStrategyId);
    setSearchParams(nextParams, { replace: true });
  };

  const handleRun = async (replayMode: 'exact' | 'custom') => {
    if (!loadedRun || !loaded) {
      setRunError(zh ? '请先加载一个历史 Run 和交易策略。' : 'Load a historical Run and trading strategy first.');
      return;
    }
    if (replayMode === 'exact' && !loadedExactReplayAvailable) {
      setRunError(zh ? '当前交易策略不是该 Run 可精确重放的原始策略。' : 'This trading strategy is not an exact-replay strategy for the Run.');
      return;
    }
    if (replayMode === 'custom' && !loadedRun.customReplayAvailable) {
      setRunError(zh ? '该 Run 缺少回测所需的数据身份或策略配置。' : 'This Run lacks the data identity or strategy contract needed for a backtest.');
      return;
    }

    const parsedTopk = Number(topk);
    const parsedCostBps = Number(costBps);
    if (replayMode === 'custom') {
      if (!startDate || !endDate || startDate > endDate) {
        setRunError(zh ? '请选择有效的回测开始和结束日期。' : 'Choose a valid backtest date range.');
        return;
      }
      if (!Number.isInteger(parsedTopk) || parsedTopk < 1 || parsedTopk > 500) {
        setRunError(zh ? '每次持有股票数必须是 1 到 500 的整数。' : 'TopK must be an integer from 1 to 500.');
        return;
      }
      if (!Number.isFinite(parsedCostBps) || parsedCostBps < 0 || parsedCostBps > 1000) {
        setRunError(zh ? '回测成本必须在 0 到 1000 基点之间。' : 'Cost must be between 0 and 1000 basis points.');
        return;
      }
    }

    setRunning(true);
    setRunError(null);
    setRunResult(null);
    const requestVersion = runRequestVersionRef.current + 1;
    runRequestVersionRef.current = requestVersion;
    try {
      const result = await backtestApi.runTrainedStrategy({
        runId: loadedRun.runId,
        modelId: loadedRun.modelId,
        strategyId: loaded.strategyId,
        replayMode,
        ...(replayMode === 'custom' ? {
          startDate,
          endDate,
          topk: parsedTopk,
          costBps: parsedCostBps,
        } : {}),
      });
      if (runRequestVersionRef.current === requestVersion) {
        setRunResult(result);
      }
    } catch (reason: unknown) {
      const parsed = getParsedApiError(reason);
      if (runRequestVersionRef.current === requestVersion) {
        setRunError(parsed.message || parsed.rawMessage || (zh ? '回测运行失败。' : 'Backtest failed.'));
      }
    } finally {
      if (runRequestVersionRef.current === requestVersion) {
        setRunning(false);
      }
    }
  };

  return (
    <main className="min-h-0 flex-1 overflow-y-auto p-3 sm:p-4">
      <div className="mx-auto max-w-[1500px] space-y-3">
        <section className="border border-border/70 bg-card/35">
          <div className="flex flex-wrap items-start justify-between gap-3 border-b border-border/70 px-4 py-3">
            <div>
              <div className="flex items-center gap-2">
                <FlaskConical className="h-4 w-4 text-cyan" />
                <h2 className="text-base font-semibold text-foreground">
                  {zh ? '选股模型与交易策略' : 'Stock-selection models and trading strategies'}
                </h2>
              </div>
              <p className="mt-1 text-xs text-secondary-text">
                {zh ? '先选择具体 Run，再选择该 Run 冻结的交易策略。' : 'Choose a specific Run, then one of the trading strategies frozen with it.'}
              </p>
            </div>
            {!loading && loadedRun ? (
              <Badge
                variant={loadedExactReplayAvailable && !loadedProviderNotice ? 'success' : 'warning'}
                size="sm"
                className="gap-1.5 shadow-none"
              >
                {loadedExactReplayAvailable && !loadedProviderNotice
                  ? <CheckCircle2 className="h-3.5 w-3.5" />
                  : <AlertTriangle className="h-3.5 w-3.5" />}
                {loadedProviderNotice?.badge
                  || (loadedExactReplayAvailable
                    ? (zh ? '支持原条件重放' : 'Exact replay available')
                    : (zh ? '仅自定义回测' : 'Custom only'))}
              </Badge>
            ) : null}
          </div>

          {error ? (
            <div className="p-4">
              <InlineAlert
                variant="danger"
                title={zh ? '历史 Run 目录加载失败' : 'Historical Run catalog failed'}
                message={error === 'load_failed'
                  ? (zh ? '后端可能仍是旧版本。请重启 DSA 服务后重新读取。' : 'The backend may be an old build. Restart DSA and retry.')
                  : error}
              />
            </div>
          ) : null}
          {!loading && selectionError ? (
            <div className="border-b border-border/70 p-4">
              <InlineAlert
                variant="danger"
                title={zh ? '无法加载链接中的回测对象' : 'Unable to load the linked backtest target'}
                message={zh
                  ? '链接指定的 Run、选股模型或交易策略不可用或彼此不匹配。请从列表中重新选择。'
                  : 'The Run, stock-selection model, or trading strategy in this link is unavailable or mismatched. Select a valid combination from the list.'}
              />
            </div>
          ) : null}
          {!loading && loadedProviderNotice ? (
            <div className="border-b border-border/70 p-4">
              <InlineAlert
                variant="warning"
                title={loadedProviderNotice.title}
                message={loadedProviderNotice.message}
              />
            </div>
          ) : null}
          {loading ? (
            <div className="flex min-h-40 items-center justify-center gap-2 text-sm text-secondary-text">
              <Loader2 className="h-4 w-4 animate-spin" />
              {zh ? '正在读取历史 Run 目录' : 'Loading historical Run catalog'}
            </div>
          ) : (
            <div className="grid gap-3 p-4 md:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)_auto] md:items-end">
              <label className="space-y-1">
                <span className="text-xs text-secondary-text">{zh ? '选股模型' : 'Stock-selection model'}</span>
                <select
                  aria-label={zh ? '选股模型' : 'Stock-selection model'}
                  value={draftRunKey}
                  onChange={(event) => handleRunChange(event.currentTarget.value)}
                  className="input-surface input-focus-glow h-10 w-full border bg-transparent px-3 text-sm focus:outline-none"
                >
                  <option value="">{zh ? '选择历史 Run' : 'Select a historical Run'}</option>
                  {selectableRuns.map((run) => (
                    <option key={makeRunKey(run)} value={makeRunKey(run)}>
                      {runOptionLabel(run, language)}
                    </option>
                  ))}
                </select>
              </label>
              <label className="space-y-1">
                <span className="text-xs text-secondary-text">{zh ? '交易策略' : 'Trading strategy'}</span>
                <select
                  aria-label={zh ? '交易策略' : 'Trading strategy'}
                  value={draftStrategyId}
                  onChange={(event) => setDraftStrategyId(event.currentTarget.value)}
                  className="input-surface input-focus-glow h-10 w-full border bg-transparent px-3 text-sm focus:outline-none"
                >
                  <option value="">{zh ? '选择交易策略' : 'Select trading strategy'}</option>
                  {draftStrategies.map((strategy) => (
                    <option key={strategy.id} value={strategy.id}>{strategyName(strategy, language)}</option>
                  ))}
                </select>
              </label>
              <button
                type="button"
                className="btn-primary h-10 gap-2 px-4 text-sm"
                disabled={!draftRun || !draftStrategyId}
                onClick={handleLoad}
              >
                <Upload className="h-4 w-4" />
                {zh ? '加载' : 'Load'}
              </button>
            </div>
          )}
          {!loading && unavailableRuns.length > 0 ? (
            <div className="border-t border-border/70 p-4">
              <InlineAlert
                variant="warning"
                title={zh
                  ? `${unavailableRuns.length} 个历史 Run 缺少回测契约`
                  : `${unavailableRuns.length} historical Run${unavailableRuns.length === 1 ? '' : 's'} excluded from backtesting`}
                message={unavailableReasons.join(zh ? '；' : ' ')}
              />
              <details className="mt-2 text-xs text-secondary-text">
                <summary className="cursor-pointer font-medium text-foreground">
                  {zh ? '查看未纳入选择列表的 Run' : 'View Runs excluded from the selector'}
                </summary>
                <ul className="mt-2 space-y-1 pl-4">
                  {unavailableRuns.slice(0, 20).map((run) => (
                    <li key={makeRunKey(run)} className="break-all">
                      {runOptionLabel(run, language)}: {zh
                        ? run.replayUnavailableReason || '缺少完整回测契约。'
                        : 'Missing immutable data identity or frozen trading-strategy contract.'}
                    </li>
                  ))}
                </ul>
                {unavailableRuns.length > 20 ? (
                  <p className="mt-2 pl-4">
                    {zh ? `另有 ${unavailableRuns.length - 20} 个未显示。` : `${unavailableRuns.length - 20} more not shown.`}
                  </p>
                ) : null}
              </details>
            </div>
          ) : null}
        </section>

        {loadedRun && loaded ? (
          <>
            <section className="grid gap-px border border-border/70 bg-border/70 sm:grid-cols-2 lg:grid-cols-4">
              <div className="bg-card px-3 py-3">
                <p className="text-xs text-muted-text">{zh ? '选股模型' : 'Stock model'}</p>
                <p className="mt-1 text-sm font-medium text-foreground">{loadedRun.modelLabel}</p>
              </div>
              <div className="bg-card px-3 py-3">
                <p className="text-xs text-muted-text">{zh ? '交易策略' : 'Trading strategy'}</p>
                <p className="mt-1 text-sm font-medium text-foreground">{strategyName(loadedCatalogStrategy, language)}</p>
              </div>
              <div className="bg-card px-3 py-3">
                <p className="text-xs text-muted-text">Run ID</p>
                <p className="mt-1 break-all font-mono text-xs text-foreground">{loadedRun.runId}</p>
              </div>
              <div className="bg-card px-3 py-3">
                <p className="text-xs text-muted-text">{zh ? '数据快照' : 'Data snapshot'}</p>
                <p className="mt-1 break-all font-mono text-xs text-foreground">
                  {loadedRun.providerIdentity.snapshotId || '-'}
                </p>
              </div>
            </section>

            {isCurrentBaseline ? (
              <section className="border border-border/70 bg-card/30 p-4">
                <h3 className="text-sm font-medium text-foreground">{zh ? '保存的历史指标（不是本次重算）' : 'Saved historical metrics (not this rerun)'}</h3>
                <div className="mt-3 grid gap-px bg-border/60 sm:grid-cols-2 lg:grid-cols-4">
                  {Object.entries(status?.latestTraining?.metrics ?? {}).map(([name, value]) => (
                    <div key={name} className="bg-card px-3 py-3">
                      <p className="truncate text-xs text-muted-text" title={name}>{name}</p>
                      <p className="mt-1 font-mono text-sm text-foreground">{formatNumber(value)}</p>
                    </div>
                  ))}
                </div>
              </section>
            ) : currentCandidateStrategy ? (
              <section className="space-y-2">
                <h3 className="text-sm font-medium text-foreground">{zh ? '保存的历史指标（用于对照）' : 'Saved historical metrics (reference)'}</h3>
                <div className="grid gap-3 xl:grid-cols-2">
                  <PeriodMetrics title={zh ? '开发期' : 'Development'} metrics={currentCandidateStrategy.selection} language={language} />
                  <PeriodMetrics title={zh ? '留出期' : 'Holdout'} metrics={currentCandidateStrategy.holdout} language={language} />
                </div>
              </section>
            ) : null}

            <section className="border border-border/70 bg-card/30 px-4 py-3">
              <div className="grid gap-3 text-xs text-secondary-text md:grid-cols-2 xl:grid-cols-4">
                {Object.entries(loadedRun.periods).map(([name, dates]) => (
                  <span key={name}>{name}: {dates.join(' - ')}</span>
                ))}
                <span className="break-all">{zh ? '配置哈希' : 'Config hash'}: {loadedRun.configHash || '-'}</span>
              </div>
              <details className="mt-3 border-t border-border/60 pt-3 text-xs text-secondary-text">
                <summary className="cursor-pointer font-medium text-foreground">{zh ? '查看冻结的交易策略配置' : 'View frozen trading-strategy configuration'}</summary>
                <pre className="mt-2 max-h-52 overflow-auto whitespace-pre-wrap break-all font-mono">
                  {JSON.stringify(loadedCatalogStrategy?.config ?? {}, null, 2)}
                </pre>
              </details>
            </section>

            {loadedRun.replayUnavailableReason ? (
              <InlineAlert
                variant="warning"
                title={zh ? '重放能力受限' : 'Replay capability limited'}
                message={loadedRun.replayUnavailableReason}
              />
            ) : null}
            {!loadedExactReplayAvailable && loadedRun.customReplayAvailable ? (
              <InlineAlert
                variant="warning"
                title={zh ? '当前交易策略仅支持自定义回测' : 'Custom backtest only for this strategy'}
                message={zh
                  ? '只有训练时实际运行并冻结了回测引擎的原始交易策略，才能称为“按原条件重放”。'
                  : 'Only the original trading strategy actually run and frozen during training can be called an exact replay.'}
              />
            ) : null}

            <section className="border border-border/70 bg-card/35 p-4">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <h3 className="text-sm font-semibold text-foreground">{zh ? '回放方式' : 'Replay mode'}</h3>
                  <p className="mt-1 text-xs leading-5 text-secondary-text">
                    {zh
                      ? loadedCatalogStrategy?.replayEngine === 'qlib_portana'
                        ? '原条件重放使用训练时相同的 Qlib 组合回测引擎，并校验保存指标；当前设置新回测使用下方参数。两者都不会重新训练选股模型。'
                        : '原条件重放使用该 Run 冻结的 DSA 交易策略引擎；当前设置新回测使用下方参数。两者都不会重新训练选股模型。'
                      : loadedCatalogStrategy?.replayEngine === 'qlib_portana'
                        ? 'Exact replay uses the same Qlib portfolio engine and validates saved metrics. A custom backtest uses the fields below. Neither retrains the model.'
                        : 'Exact replay uses the Run\'s frozen DSA strategy engine. A custom backtest uses the fields below. Neither retrains the model.'}
                  </p>
                </div>
                <span className="border border-border/70 bg-card/60 px-2 py-1 font-mono text-xs text-secondary-text">{loadedRun.runId}</span>
              </div>

              <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                <label className="space-y-1">
                  <span className="text-xs text-secondary-text">{zh ? '回测开始日期' : 'Backtest start date'}</span>
                  <input
                    type="date"
                    aria-label={zh ? '回测开始日期' : 'Backtest start date'}
                    value={startDate}
                    onChange={(event) => setStartDate(event.currentTarget.value)}
                    disabled={running}
                    className="input-surface input-focus-glow h-10 w-full border bg-transparent px-3 text-sm focus:outline-none"
                  />
                </label>
                <label className="space-y-1">
                  <span className="text-xs text-secondary-text">{zh ? '回测结束日期' : 'Backtest end date'}</span>
                  <input
                    type="date"
                    aria-label={zh ? '回测结束日期' : 'Backtest end date'}
                    value={endDate}
                    onChange={(event) => setEndDate(event.currentTarget.value)}
                    disabled={running}
                    className="input-surface input-focus-glow h-10 w-full border bg-transparent px-3 text-sm focus:outline-none"
                  />
                </label>
                <label className="space-y-1">
                  <span className="text-xs text-secondary-text">{zh ? '每次持有股票数（TopK）' : 'Stocks held each time (TopK)'}</span>
                  <input
                    type="number"
                    min="1"
                    max="500"
                    step="1"
                    aria-label={zh ? '每次持有股票数（TopK）' : 'Stocks held each time (TopK)'}
                    value={topk}
                    onChange={(event) => setTopk(event.currentTarget.value)}
                    disabled={running}
                    className="input-surface input-focus-glow h-10 w-full border bg-transparent px-3 text-sm focus:outline-none"
                  />
                </label>
                <label className="space-y-1">
                  <span className="text-xs text-secondary-text">{zh ? '回测成本（基点，1基点=0.01%）' : 'Backtest cost (1 bp = 0.01%)'}</span>
                  <input
                    type="number"
                    min="0"
                    max="1000"
                    step="1"
                    aria-label={zh ? '回测成本（基点，1基点=0.01%）' : 'Backtest cost (1 bp = 0.01%)'}
                    value={costBps}
                    onChange={(event) => setCostBps(event.currentTarget.value)}
                    disabled={running}
                    className="input-surface input-focus-glow h-10 w-full border bg-transparent px-3 text-sm focus:outline-none"
                  />
                </label>
              </div>

              <div className="mt-4 flex flex-col gap-3 border-t border-border/60 pt-3 lg:flex-row lg:items-center lg:justify-between">
                <p className="text-xs leading-5 text-secondary-text">
                  {zh
                    ? `${costBps || '0'} 基点表示：发生 100% 换手时扣除 ${(Number(costBps || 0) / 100).toFixed(2)}% 成本。`
                    : `${costBps || '0'} bps means ${(Number(costBps || 0) / 100).toFixed(2)}% cost at 100% turnover.`}
                </p>
                <div className="flex flex-col gap-2 sm:flex-row">
                  <button
                    type="button"
                    onClick={() => void handleRun('exact')}
                    disabled={running || !loadedExactReplayAvailable}
                    className="btn-secondary h-10 gap-2 px-4 text-sm"
                  >
                    {running ? <Loader2 className="h-4 w-4 animate-spin" /> : <RotateCcw className="h-4 w-4" />}
                    {zh ? '按原条件重放' : 'Replay original conditions'}
                  </button>
                  <button
                    type="button"
                    onClick={() => void handleRun('custom')}
                    disabled={running || !loadedRun.customReplayAvailable}
                    className="btn-primary h-10 gap-2 px-4 text-sm"
                  >
                    {running ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                    {zh ? '按当前设置新回测' : 'Run custom backtest'}
                  </button>
                </div>
              </div>
            </section>

            {runError ? <InlineAlert variant="danger" title={zh ? '回测失败' : 'Backtest failed'} message={runError} /> : null}
            {runResult ? <TrainedStrategyBacktestResults language={language} report={runResult} /> : null}
          </>
        ) : !loading && !error ? (
          <div className="flex items-center gap-2 border border-warning/30 bg-warning/5 px-4 py-3 text-sm text-secondary-text">
            <AlertTriangle className="h-4 w-4 text-warning" />
            {zh ? '当前没有可加载的正式选股模型 Run。测试结果不会出现在这里。' : 'There are no formal trained Runs to load. Test results are excluded.'}
          </div>
        ) : null}
      </div>
    </main>
  );
};

export default ModelStrategyBacktestPanel;
