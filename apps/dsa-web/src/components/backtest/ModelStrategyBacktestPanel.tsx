import type React from 'react';
import { useEffect, useRef, useState } from 'react';
import { AlertTriangle, CheckCircle2, FlaskConical, Loader2, Play, Upload } from 'lucide-react';
import { useSearchParams } from 'react-router-dom';
import { backtestApi } from '../../api/backtest';
import { getParsedApiError } from '../../api/error';
import {
  researchApi,
  type ResearchCandidateMetrics,
  type ResearchCandidateModel,
  type ResearchCandidateStrategy,
  type ResearchStatus,
} from '../../api/research';
import { Badge, InlineAlert } from '../common';
import type { TrainedStrategyBacktestResponse } from '../../types/backtest';
import TrainedStrategyBacktestResults from './TrainedStrategyBacktestResults';

type ModelStrategyBacktestPanelProps = {
  language: 'zh' | 'en';
};

type LoadedSelection = {
  modelId: string;
  strategyId: string;
  runId?: string;
};

type RequestedSelection = {
  modelId: string | null;
  strategyId: string | null;
  runId: string | null;
};

const STRATEGY_LABELS: Record<string, { zh: string; en: string }> = {
  qlib_default: { zh: '每日 TopK 等权基线', en: 'Daily equal-weight TopK baseline' },
  baseline_daily: { zh: '每日 TopK 等权基线', en: 'Daily equal-weight TopK baseline' },
  low_turnover_weekly: { zh: '每周低换手', en: 'Weekly low turnover' },
  ma200_weekly: { zh: '200日均线周频', en: 'MA200 weekly' },
  volatility_weekly: { zh: '波动率控制周频', en: 'Volatility-controlled weekly' },
  combined_risk: { zh: '完整组合风控', en: 'Combined risk control' },
};

const TRAINED_MODEL_STRATEGY_IDS = [
  'baseline_daily',
  'low_turnover_weekly',
  'ma200_weekly',
  'volatility_weekly',
  'combined_risk',
] as const;

function normalizeStrategyId(strategyId: string | null): string | null {
  return strategyId === 'qlib_default' ? 'baseline_daily' : strategyId;
}

function trainedModelStrategies(language: 'zh' | 'en'): ResearchCandidateStrategy[] {
  return TRAINED_MODEL_STRATEGY_IDS.map((strategyId) => ({
    modelId: 'latest',
    strategyId,
    strategyLabel: STRATEGY_LABELS[strategyId][language],
  }));
}

function formatPercent(value?: number | null): string {
  return typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(2)}%` : '-';
}

function formatNumber(value?: number | null): string {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(2) : '-';
}

function modelRunId(status: ResearchStatus, modelId: string): string | undefined {
  if (modelId === 'latest') return status.latestTraining?.runId;
  return status.candidatePipeline?.models?.find((model) => model.id === modelId)?.runId || undefined;
}

function strategyName(row: ResearchCandidateStrategy | undefined, strategyId: string, language: 'zh' | 'en'): string {
  return STRATEGY_LABELS[strategyId]?.[language] || row?.strategyLabel || strategyId;
}

function initialSelection(status: ResearchStatus, requested: RequestedSelection): LoadedSelection | null {
  const models = status.candidatePipeline?.models ?? [];
  const strategies = status.candidatePipeline?.strategyEvaluations ?? [];
  const requestedStrategyId = normalizeStrategyId(requested.strategyId);
  const requestedIsValid = requested.modelId === 'latest'
    ? Boolean(status.latestTraining
      && requestedStrategyId
      && TRAINED_MODEL_STRATEGY_IDS.includes(requestedStrategyId as typeof TRAINED_MODEL_STRATEGY_IDS[number]))
    : Boolean(models.some((model) => model.id === requested.modelId)
      && strategies.some((row) => row.modelId === requested.modelId && row.strategyId === requestedStrategyId));

  if (requested.modelId && requestedStrategyId && requestedIsValid) {
    return {
      modelId: requested.modelId,
      strategyId: requestedStrategyId,
      runId: requested.runId || modelRunId(status, requested.modelId),
    };
  }

  const preferred = status.candidatePipeline?.selectedCombination
    ?? status.candidatePipeline?.partialDevelopmentLeader
    ?? strategies[0];
  if (preferred) {
    return {
      modelId: preferred.modelId,
      strategyId: preferred.strategyId,
      runId: modelRunId(status, preferred.modelId),
    };
  }
  if (status.latestTraining) {
    return { modelId: 'latest', strategyId: 'baseline_daily', runId: status.latestTraining.runId };
  }
  return null;
}

function defaultBacktestRange(status: ResearchStatus): { start: string; end: string } {
  const periods = status.candidatePipeline?.periods;
  const selection = periods?.selection ?? [];
  const holdout = periods?.holdout ?? [];
  return {
    start: selection[0] || holdout[0] || '',
    end: holdout[1] || selection[1] || '',
  };
}

const MetricCell: React.FC<{ label: string; value: string; tone?: 'positive' | 'negative' | 'neutral' }> = ({
  label,
  value,
  tone = 'neutral',
}) => (
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
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [draftModelId, setDraftModelId] = useState(searchParams.get('modelId') || '');
  const [draftStrategyId, setDraftStrategyId] = useState(searchParams.get('strategyId') || '');
  const [loaded, setLoaded] = useState<LoadedSelection | null>(() => {
    const modelId = searchParams.get('modelId');
    const strategyId = searchParams.get('strategyId');
    if (!modelId || !strategyId) return null;
    return { modelId, strategyId, runId: searchParams.get('runId') || undefined };
  });
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [topk, setTopk] = useState('50');
  const [costBps, setCostBps] = useState('15');
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [runResult, setRunResult] = useState<TrainedStrategyBacktestResponse | null>(null);
  const requestedSelectionRef = useRef<RequestedSelection>({
    modelId: searchParams.get('modelId'),
    strategyId: searchParams.get('strategyId'),
    runId: searchParams.get('runId'),
  });

  useEffect(() => {
    let active = true;
    researchApi.getStatus()
      .then((payload) => {
        if (!active) return;
        const next = initialSelection(payload, requestedSelectionRef.current);
        setStatus(payload);
        setError(null);
        if (next) {
          const range = defaultBacktestRange(payload);
          setDraftModelId(next.modelId);
          setDraftStrategyId(next.strategyId);
          setLoaded(next);
          setStartDate(range.start);
          setEndDate(range.end);
        }
      })
      .catch((reason: unknown) => {
        if (!active) return;
        const parsed = getParsedApiError(reason);
        setError(parsed.message || parsed.rawMessage || 'load_failed');
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  const models = status?.candidatePipeline?.models ?? [];
  const strategies = status?.candidatePipeline?.strategyEvaluations ?? [];
  const modelOptions: Array<{ id: string; label: string; runId?: string }> = status?.latestTraining
    ? [{
        id: 'latest',
        label: status.latestTraining.experimentName || (zh ? '最近训练结果' : 'Latest training result'),
        runId: status.latestTraining.runId,
      }]
    : [];
  for (const model of models) {
    modelOptions.push({ id: model.id, label: model.label || model.id, runId: model.runId || undefined });
  }
  const draftStrategies: ResearchCandidateStrategy[] = draftModelId === 'latest'
    ? trainedModelStrategies(language)
    : strategies.filter((row) => row.modelId === draftModelId);

  const loadedModel: ResearchCandidateModel | undefined = loaded?.modelId === 'latest'
    ? undefined
    : models.find((model) => model.id === loaded?.modelId);
  const loadedStrategy = strategies.find(
    (row) => row.modelId === loaded?.modelId && row.strategyId === loaded?.strategyId,
  );
  const loadedModelLabel = loaded?.modelId === 'latest'
    ? status?.latestTraining?.experimentName || (zh ? '最近训练结果' : 'Latest training result')
    : loadedModel?.label || loaded?.modelId || '-';
  const loadedStrategyLabel = loaded
    ? strategyName(loadedStrategy, loaded.strategyId, language)
    : '-';
  const periods = status?.candidatePipeline?.periods;

  const handleModelChange = (modelId: string) => {
    setDraftModelId(modelId);
    if (modelId === 'latest') {
      setDraftStrategyId('baseline_daily');
      return;
    }
    setDraftStrategyId(strategies.find((row) => row.modelId === modelId)?.strategyId || '');
  };

  const handleLoad = () => {
    if (!status || !draftModelId || !draftStrategyId) return;
    const next = {
      modelId: draftModelId,
      strategyId: draftStrategyId,
      runId: modelRunId(status, draftModelId),
    };
    setLoaded(next);
    const range = defaultBacktestRange(status);
    setStartDate(range.start);
    setEndDate(range.end);
    setRunResult(null);
    setRunError(null);
    const nextParams = new URLSearchParams(searchParams);
    nextParams.set('tab', 'trained');
    if (next.runId) nextParams.set('runId', next.runId);
    else nextParams.delete('runId');
    nextParams.set('modelId', next.modelId);
    nextParams.set('strategyId', next.strategyId);
    setSearchParams(nextParams, { replace: true });
  };

  const handleRun = async () => {
    const runId = loaded?.runId || (loaded && status ? modelRunId(status, loaded.modelId) : undefined);
    const parsedTopk = Number(topk);
    const parsedCostBps = Number(costBps);
    if (!loaded || !runId) {
      setRunError(zh ? '当前结果没有可用的 Run ID，不能重新回测。' : 'This result has no usable Run ID.');
      return;
    }
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

    setRunning(true);
    setRunError(null);
    setRunResult(null);
    try {
      setRunResult(await backtestApi.runTrainedStrategy({
        runId,
        modelId: loaded.modelId,
        strategyId: loaded.strategyId,
        startDate,
        endDate,
        topk: parsedTopk,
        costBps: parsedCostBps,
      }));
    } catch (reason: unknown) {
      const parsed = getParsedApiError(reason);
      setRunError(parsed.message || parsed.rawMessage || (zh ? '回测运行失败。' : 'Backtest failed.'));
    } finally {
      setRunning(false);
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
                {zh ? '选择选股模型和交易策略，再设置日期、持股数量和成本。' : 'Choose a stock-selection model and trading strategy, then set dates, holdings and cost.'}
              </p>
            </div>
            {!loading && status && loaded ? (
              <Badge variant="success" size="sm" className="gap-1.5 shadow-none">
                <CheckCircle2 className="h-3.5 w-3.5" />
                {zh ? '选股模型已加载' : 'Stock-selection model loaded'}
              </Badge>
            ) : null}
          </div>

          {error ? (
            <div className="p-4">
              <InlineAlert
                variant="danger"
                title={zh ? '加载失败' : 'Load failed'}
                message={error === 'load_failed' ? (zh ? '训练结果加载失败。' : 'Failed to load training results.') : error}
              />
            </div>
          ) : null}
          {loading ? (
            <div className="flex min-h-40 items-center justify-center gap-2 text-sm text-secondary-text">
              <Loader2 className="h-4 w-4 animate-spin" />
              {zh ? '正在读取训练结果' : 'Loading training results'}
            </div>
          ) : (
            <div className="grid gap-3 p-4 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto] md:items-end">
              <label className="space-y-1">
                <span className="text-xs text-secondary-text">{zh ? '选股模型' : 'Stock-selection model'}</span>
                <select
                  aria-label={zh ? '选股模型' : 'Stock-selection model'}
                  value={draftModelId}
                  onChange={(event) => handleModelChange(event.currentTarget.value)}
                  className="input-surface input-focus-glow h-10 w-full border bg-transparent px-3 text-sm focus:outline-none"
                >
                  <option value="">{zh ? '选择选股模型' : 'Select stock-selection model'}</option>
                  {modelOptions.map((model) => (
                    <option key={model.id} value={model.id}>{model.label}</option>
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
                  {draftStrategies.map((row) => (
                    <option key={row.strategyId} value={row.strategyId}>
                      {strategyName(row, row.strategyId, language)}
                    </option>
                  ))}
                </select>
              </label>
              <button
                type="button"
                className="btn-primary h-10 gap-2 px-4 text-sm"
                disabled={!draftModelId || !draftStrategyId}
                onClick={handleLoad}
              >
                <Upload className="h-4 w-4" />
                {zh ? '加载选股模型与交易策略' : 'Load stock-selection model and trading strategy'}
              </button>
            </div>
          )}
        </section>

        {loaded && status ? (
          <>
            <section className="grid gap-px border border-border/70 bg-border/70 sm:grid-cols-2 lg:grid-cols-4">
              <div className="bg-card px-3 py-3">
                <p className="text-xs text-muted-text">{zh ? '选股模型' : 'Stock model'}</p>
                <p className="mt-1 text-sm font-medium text-foreground">{loadedModelLabel}</p>
              </div>
              <div className="bg-card px-3 py-3">
                <p className="text-xs text-muted-text">{zh ? '交易策略' : 'Trading strategy'}</p>
                <p className="mt-1 text-sm font-medium text-foreground">{loadedStrategyLabel}</p>
              </div>
              <div className="bg-card px-3 py-3">
                <p className="text-xs text-muted-text">Run ID</p>
                <p className="mt-1 break-all font-mono text-xs text-foreground">{loaded.runId || modelRunId(status, loaded.modelId) || '-'}</p>
              </div>
              <div className="bg-card px-3 py-3">
                <p className="text-xs text-muted-text">{zh ? '数据版本' : 'Data version'}</p>
                <p className="mt-1 text-sm text-foreground">
                  {loaded.modelId === 'latest' ? status.latestTraining?.provider : status.candidatePipeline?.provider || '-'}
                </p>
              </div>
            </section>

            {loaded.modelId === 'latest' ? (
              <section className="border border-border/70 bg-card/30 p-4">
                <h3 className="text-sm font-medium text-foreground">{zh ? '保存的历史指标（不是本次重算）' : 'Saved historical metrics (not this rerun)'}</h3>
                <div className="mt-3 grid gap-px bg-border/60 sm:grid-cols-2 lg:grid-cols-4">
                  {Object.entries(status.latestTraining?.metrics ?? {}).map(([name, value]) => (
                    <div key={name} className="bg-card px-3 py-3">
                      <p className="truncate text-xs text-muted-text" title={name}>{name}</p>
                      <p className="mt-1 font-mono text-sm text-foreground">{formatNumber(value)}</p>
                    </div>
                  ))}
                </div>
              </section>
            ) : loadedStrategy ? (
              <section className="space-y-2">
                <h3 className="text-sm font-medium text-foreground">{zh ? '保存的历史指标（用于对照）' : 'Saved historical metrics (reference)'}</h3>
                <div className="grid gap-3 xl:grid-cols-2">
                  <PeriodMetrics title={zh ? '开发期' : 'Development'} metrics={loadedStrategy.selection} language={language} />
                  <PeriodMetrics title={zh ? '留出期' : 'Holdout'} metrics={loadedStrategy.holdout} language={language} />
                </div>
              </section>
            ) : (
              <InlineAlert
                variant="warning"
                title={zh ? '交易策略结果缺失' : 'Trading-strategy result missing'}
                message={zh ? '链接中的选股模型或交易策略不在当前候选结果中。' : 'The linked stock-selection model or trading strategy is not present in the current candidate results.'}
              />
            )}

            {periods ? (
              <section className="flex flex-wrap items-center gap-x-5 gap-y-2 border border-border/70 bg-card/30 px-4 py-3 text-xs text-secondary-text">
                <span>{zh ? '训练期' : 'Training'}: {(periods.training ?? []).join(' - ') || '-'}</span>
                <span>{zh ? '开发期' : 'Development'}: {(periods.selection ?? []).join(' - ') || '-'}</span>
                <span>{zh ? '留出期' : 'Holdout'}: {(periods.holdout ?? []).join(' - ') || '-'}</span>
              </section>
            ) : null}

            <section className="border border-border/70 bg-card/35 p-4">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <h3 className="text-sm font-semibold text-foreground">
                    {zh ? '设置本次回测' : 'Configure this backtest'}
                  </h3>
                  <p className="mt-1 text-xs leading-5 text-secondary-text">
                    {zh
                      ? '读取这个 Run 已保存的选股模型评分和真实结果，按所选交易策略重新计算；不会重新训练选股模型。'
                      : 'Uses this Run\'s saved stock-selection scores and realized labels to recalculate the trading strategy; it does not retrain the stock-selection model.'}
                  </p>
                </div>
                <span className="border border-border/70 bg-card/60 px-2 py-1 font-mono text-xs text-secondary-text">
                  {loaded.runId || modelRunId(status, loaded.modelId) || '-'}
                </span>
              </div>

              <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                <label className="space-y-1">
                  <span className="text-xs text-secondary-text">{zh ? '回测开始日期' : 'Backtest start date'}</span>
                  <input
                    type="date"
                    aria-label={zh ? '回测开始日期' : 'Backtest start date'}
                    value={startDate}
                    onChange={event => setStartDate(event.currentTarget.value)}
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
                    onChange={event => setEndDate(event.currentTarget.value)}
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
                    onChange={event => setTopk(event.currentTarget.value)}
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
                    onChange={event => setCostBps(event.currentTarget.value)}
                    disabled={running}
                    className="input-surface input-focus-glow h-10 w-full border bg-transparent px-3 text-sm focus:outline-none"
                  />
                </label>
              </div>

              <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-border/60 pt-3">
                <p className="text-xs leading-5 text-secondary-text">
                  {zh
                    ? `${costBps || '0'} 基点表示：发生 100% 换手时扣除 ${(Number(costBps || 0) / 100).toFixed(2)}% 成本；TopK 是每次按选股模型评分持有的股票数量。`
                    : `${costBps || '0'} bps means ${(Number(costBps || 0) / 100).toFixed(2)}% cost at 100% turnover; TopK is the number of highest-scored stocks held.`}
                </p>
                <button
                  type="button"
                  onClick={() => void handleRun()}
                  disabled={running || !loaded.runId}
                  className="btn-primary h-10 min-w-[120px] gap-2 px-4 text-sm"
                >
                  {running ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                  {running ? (zh ? '正在回测' : 'Running') : (zh ? '运行回测' : 'Run backtest')}
                </button>
              </div>
            </section>

            {runError ? <InlineAlert variant="danger" title={zh ? '回测失败' : 'Backtest failed'} message={runError} /> : null}
            {runResult ? <TrainedStrategyBacktestResults language={language} report={runResult} /> : null}
          </>
        ) : !loading && !error ? (
          <div className="flex items-center gap-2 border border-warning/30 bg-warning/5 px-4 py-3 text-sm text-secondary-text">
            <AlertTriangle className="h-4 w-4 text-warning" />
            {zh ? '当前没有可加载的选股模型与交易策略结果。' : 'There are no stock-selection model and trading-strategy results to load.'}
          </div>
        ) : null}
      </div>
    </main>
  );
};

export default ModelStrategyBacktestPanel;
