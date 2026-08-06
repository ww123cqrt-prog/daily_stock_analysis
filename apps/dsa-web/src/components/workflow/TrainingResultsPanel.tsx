import type React from 'react';
import { BarChart3, CheckCircle2, FlaskConical } from 'lucide-react';
import { Link } from 'react-router-dom';
import type {
  ResearchCandidatePipeline,
  ResearchCandidateStrategy,
  ResearchTrainingResult,
} from '../../api/research';
import { providerStatusNotice } from '../../utils/providerStatus';
import { buildMlflowRunUiUrl } from '../../utils/mlflowUiUrl';
import { Badge, InlineAlert } from '../common';

type TrainingResultsPanelProps = {
  latestTraining?: ResearchTrainingResult | null;
  candidatePipeline?: ResearchCandidatePipeline | null;
  language: 'zh' | 'en';
  mlflowUrl?: string;
};

const STRATEGY_LABELS: Record<string, { zh: string; en: string }> = {
  baseline_daily: { zh: '每日 TopK 等权基线', en: 'Daily equal-weight TopK baseline' },
  low_turnover_weekly: { zh: '每周低换手', en: 'Weekly low turnover' },
  ma200_weekly: { zh: '200日均线周频', en: 'MA200 weekly' },
  volatility_weekly: { zh: '波动率控制周频', en: 'Volatility-controlled weekly' },
  combined_risk: { zh: '完整组合风控', en: 'Combined risk control' },
};

function backtestHref(
  runId: string | null | undefined,
  modelId: string,
  strategyId: string,
  snapshotId?: string,
): string {
  const params = new URLSearchParams({ tab: 'trained' });
  if (runId) params.set('runId', runId);
  params.set('modelId', modelId);
  params.set('strategyId', strategyId);
  if (snapshotId) params.set('snapshotId', snapshotId);
  return `/backtest?${params.toString()}`;
}

type BacktestEligibility = {
  ready: boolean;
  reasons: string[];
};

const SHA256_PATTERN = /^[0-9a-f]{64}$/;

function hasReplayArtifacts(artifacts?: Record<string, string>): boolean {
  return Boolean(
    artifacts
    && SHA256_PATTERN.test(artifacts['pred.pkl'] || '')
    && SHA256_PATTERN.test(artifacts['label.pkl'] || ''),
  );
}

function hasSnapshot(identity?: ResearchTrainingResult['providerIdentity']): boolean {
  return Boolean(
    identity?.providerContract === 'immutable_snapshot_v2'
    && identity.datasetId
    && identity.snapshotId
    && SHA256_PATTERN.test(identity.manifestSha256 || '')
    && SHA256_PATTERN.test(identity.contentSha256 || ''),
  );
}

function eligibility(
  input: {
    dryRun?: boolean;
    runId?: string | null;
    experimentId?: string | null;
    artifacts?: Record<string, string>;
    providerIdentity?: ResearchTrainingResult['providerIdentity'];
    strategyConfigReady: boolean;
  },
  language: 'zh' | 'en',
): BacktestEligibility {
  const zh = language === 'zh';
  const reasons: string[] = [];
  if (input.dryRun) reasons.push(zh ? '测试式训练没有真实训练产物' : 'Synthetic preview has no real training artifacts');
  if (!input.runId) reasons.push(zh ? '缺少真实 Run ID' : 'Missing real Run ID');
  if (!input.experimentId) reasons.push(zh ? '缺少实验 ID' : 'Missing experiment ID');
  if (!hasReplayArtifacts(input.artifacts)) reasons.push(zh ? '缺少可重放的预测和标签' : 'Missing replayable predictions and labels');
  if (!hasSnapshot(input.providerIdentity)) reasons.push(zh ? '缺少不可变数据快照' : 'Missing immutable data snapshot');
  if (!input.strategyConfigReady) reasons.push(zh ? '缺少交易策略配置' : 'Missing trading-strategy configuration');
  return { ready: reasons.length === 0, reasons };
}

function percent(value?: number | null): string {
  return typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(2)}%` : '-';
}

function strategyLabel(row: ResearchCandidateStrategy, language: 'zh' | 'en'): string {
  return STRATEGY_LABELS[row.strategyId]?.[language] || row.strategyLabel || row.strategyId;
}

const TrainingResultsPanel: React.FC<TrainingResultsPanelProps> = ({
  latestTraining,
  candidatePipeline,
  language,
  mlflowUrl = 'http://127.0.0.1:6006/',
}) => {
  const zh = language === 'zh';
  const models = candidatePipeline?.models ?? [];
  const strategyRows = candidatePipeline?.strategyEvaluations ?? [];
  const selected = candidatePipeline?.selectedCombination ?? candidatePipeline?.partialDevelopmentLeader;
  const baselineName = latestTraining?.experimentName || (zh ? 'Qlib 单一选股模型基线' : 'Qlib single stock-selection model baseline');
  const baselineStrategy = zh ? '每日 TopK 等权基线' : 'Daily equal-weight TopK baseline';
  const hasBaselineResult = Boolean(latestTraining);
  const hasResults = hasBaselineResult || strategyRows.length > 0;
  const latestTrainingNotice = providerStatusNotice(latestTraining, 'training', language);
  const candidateNotice = providerStatusNotice(candidatePipeline, 'candidate', language);

  return (
    <section className="overflow-hidden border border-border/70 bg-card/35" aria-labelledby="training-results-title">
      <div className="border-b border-border/70 px-4 py-3">
        <div>
          <div className="flex items-center gap-2">
            <FlaskConical className="h-4 w-4 text-cyan" />
            <h2 id="training-results-title" className="text-base font-semibold text-foreground">
              {zh ? '可回测的选股模型与交易策略' : 'Backtest-ready stock-selection models and trading strategies'}
            </h2>
          </div>
          <p className="mt-1 text-xs text-secondary-text">
            {zh
              ? '这里只列出可送入回测的结果；Run ID 是对应训练运行的编号。'
              : 'Only results ready for backtesting are listed here; the Run ID identifies the corresponding training run.'}
          </p>
        </div>
      </div>

      {latestTrainingNotice ? (
        <div className="border-b border-border/70 p-4">
          <InlineAlert
            variant="warning"
            title={latestTrainingNotice.title}
            message={latestTrainingNotice.message}
          />
        </div>
      ) : null}

      {candidateNotice ? (
        <div className="border-b border-border/70 p-4">
          <InlineAlert
            variant="warning"
            title={candidateNotice.title}
            message={candidateNotice.message}
          />
        </div>
      ) : null}

      {candidatePipeline?.dryRun ? (
        <div className="border-b border-border/70 p-4">
          <InlineAlert
            variant="warning"
            title={zh ? '测试数据，不可用于研究或交易' : 'Test data, not for research or trading'}
            message={zh
              ? '这是合成小样本预演，没有真实 Run ID、预测产物或 MLflow 训练记录。'
              : 'This synthetic preview has no real Run ID, prediction artifacts, or MLflow training record.'}
          />
        </div>
      ) : null}

      {hasResults ? (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[780px] text-left text-sm">
            <thead className="bg-surface/50 text-xs text-muted-text">
              <tr>
                <th className="px-4 py-2.5 font-medium">{zh ? '选股模型' : 'Stock model'}</th>
                <th className="px-4 py-2.5 font-medium">{zh ? '交易策略' : 'Trading strategy'}</th>
                <th className="px-4 py-2.5 font-medium">{zh ? '开发期年化超额' : 'Development annual excess'}</th>
                <th className="px-4 py-2.5 font-medium">{zh ? '留出期年化超额' : 'Holdout annual excess'}</th>
                <th className="px-4 py-2.5 font-medium">Run ID</th>
                <th className="px-4 py-2.5 text-right font-medium">{zh ? '操作' : 'Action'}</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/60">
              {hasBaselineResult ? (
                (() => {
                  const resultEligibility = eligibility({
                    runId: latestTraining?.runId,
                    experimentId: latestTraining?.experimentId,
                    artifacts: latestTraining?.artifactSha256,
                    providerIdentity: latestTraining?.providerIdentity,
                    strategyConfigReady: true,
                  }, language);
                  return (
                <tr className="hover:bg-hover/45">
                  <td className="px-4 py-3 font-medium text-foreground">{baselineName}</td>
                  <td className="px-4 py-3 text-secondary-text">
                    <span className="inline-flex items-center gap-2">
                      {baselineStrategy}
                      <Badge variant="default" size="sm" className="shadow-none">
                        {zh ? '单一选股模型基线' : 'Single stock-selection model baseline'}
                      </Badge>
                    </span>
                  </td>
                  <td className="px-4 py-3 font-mono text-muted-text">-</td>
                  <td className="px-4 py-3 font-mono text-muted-text">-</td>
                  <td className="max-w-[12rem] truncate px-4 py-3 font-mono text-xs text-secondary-text">
                    {latestTraining?.runId ? (
                      <a
                        href={buildMlflowRunUiUrl(mlflowUrl, latestTraining.experimentId, latestTraining.runId)}
                        target="_blank"
                        rel="noreferrer"
                        className="text-cyan hover:underline"
                        aria-label={`${zh ? '在 MLflow 中打开 Run' : 'Open MLflow Run'} ${latestTraining.runId}`}
                      >
                        {latestTraining.runId}
                      </a>
                    ) : '-'}
                  </td>
                  <td className="px-4 py-3 text-right">
                    {resultEligibility.ready ? (
                      <Link
                        to={backtestHref(
                          latestTraining?.runId,
                          'latest',
                          'baseline_daily',
                          latestTraining?.providerIdentity?.snapshotId,
                        )}
                        className="btn-secondary h-8 gap-1.5 px-2.5 text-xs"
                        aria-label={`${zh ? '回测' : 'Backtest'} ${baselineName} ${baselineStrategy}`}
                      >
                        <BarChart3 className="h-3.5 w-3.5" />
                        {zh ? '回测' : 'Backtest'}
                      </Link>
                    ) : (
                      <div className="flex flex-col items-end gap-1">
                        <a href="#recent-quant-jobs" className="text-xs font-medium text-cyan hover:underline">
                          {zh ? '查看测试结果' : 'View test result'}
                        </a>
                        <span className="max-w-56 text-right text-[11px] leading-4 text-muted-text">
                          {resultEligibility.reasons.join(zh ? '；' : '; ')}
                        </span>
                      </div>
                    )}
                  </td>
                </tr>
                  );
                })()
              ) : null}
              {strategyRows.map((row) => {
                const model = models.find((item) => item.id === row.modelId);
                const label = strategyLabel(row, language);
                const isSelected = selected?.modelId === row.modelId && selected?.strategyId === row.strategyId;
                const resultEligibility = eligibility({
                  dryRun: candidatePipeline?.dryRun,
                  runId: model?.runId,
                  experimentId: model?.experimentId,
                  artifacts: model?.artifactSha256,
                  providerIdentity: candidatePipeline?.providerIdentity,
                  strategyConfigReady: Boolean(row.strategyConfig),
                }, language);
                return (
                  <tr key={`${row.modelId}-${row.strategyId}`} className="hover:bg-hover/45">
                    <td className="px-4 py-3 font-medium text-foreground">{model?.label || row.modelId}</td>
                    <td className="px-4 py-3 text-secondary-text">
                      <span className="inline-flex items-center gap-2">
                        {label}
                        {isSelected ? (
                          <Badge variant="success" size="sm" className="gap-1 shadow-none">
                            <CheckCircle2 className="h-3 w-3" />
                            {zh ? '当前选择' : 'Selected'}
                          </Badge>
                        ) : null}
                      </span>
                    </td>
                    <td className="px-4 py-3 font-mono text-foreground">{percent(row.selection?.annualExcessReturn)}</td>
                    <td className="px-4 py-3 font-mono text-foreground">{percent(row.holdout?.annualExcessReturn)}</td>
                    <td className="max-w-[12rem] truncate px-4 py-3 font-mono text-xs text-secondary-text">
                      {model?.runId ? (
                        <a
                          href={buildMlflowRunUiUrl(mlflowUrl, model.experimentId, model.runId)}
                          target="_blank"
                          rel="noreferrer"
                          className="text-cyan hover:underline"
                          aria-label={`${zh ? '在 MLflow 中打开 Run' : 'Open MLflow Run'} ${model.runId}`}
                        >
                          {model.runId}
                        </a>
                      ) : '-'}
                    </td>
                    <td className="px-4 py-3 text-right">
                      {resultEligibility.ready ? (
                        <Link
                          to={backtestHref(
                            model?.runId,
                            row.modelId,
                            row.strategyId,
                            candidatePipeline?.providerIdentity?.snapshotId,
                          )}
                          className="btn-secondary h-8 gap-1.5 px-2.5 text-xs"
                          aria-label={`${zh ? '回测' : 'Backtest'} ${model?.label || row.modelId} ${label}`}
                        >
                          <BarChart3 className="h-3.5 w-3.5" />
                          {zh ? '回测' : 'Backtest'}
                        </Link>
                      ) : (
                        <div className="flex flex-col items-end gap-1">
                          <a href="#recent-quant-jobs" className="text-xs font-medium text-cyan hover:underline">
                            {candidatePipeline?.dryRun
                              ? (zh ? '查看测试结果' : 'View test result')
                              : (zh ? '查看结果' : 'View result')}
                          </a>
                          <span className="max-w-56 text-right text-[11px] leading-4 text-muted-text">
                            {resultEligibility.reasons.join(zh ? '；' : '; ')}
                          </span>
                        </div>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="px-4 py-4 text-sm text-secondary-text">
          {zh ? '尚无可送入回测的选股模型或交易策略结果。' : 'No stock-selection model or trading-strategy result is ready for backtesting yet.'}
        </p>
      )}
    </section>
  );
};

export default TrainingResultsPanel;
