import type React from 'react';
import { BarChart3, CheckCircle2, FlaskConical } from 'lucide-react';
import { Link } from 'react-router-dom';
import type {
  ResearchCandidatePipeline,
  ResearchCandidateStrategy,
  ResearchTrainingResult,
} from '../../api/research';
import { Badge } from '../common';

type TrainingResultsPanelProps = {
  latestTraining?: ResearchTrainingResult | null;
  candidatePipeline?: ResearchCandidatePipeline | null;
  language: 'zh' | 'en';
};

const STRATEGY_LABELS: Record<string, { zh: string; en: string }> = {
  baseline_daily: { zh: '每日 TopK 等权基线', en: 'Daily equal-weight TopK baseline' },
  low_turnover_weekly: { zh: '每周低换手', en: 'Weekly low turnover' },
  ma200_weekly: { zh: '200日均线周频', en: 'MA200 weekly' },
  volatility_weekly: { zh: '波动率控制周频', en: 'Volatility-controlled weekly' },
  combined_risk: { zh: '完整组合风控', en: 'Combined risk control' },
};

function backtestHref(runId: string | null | undefined, modelId: string, strategyId: string): string {
  const params = new URLSearchParams({ tab: 'trained' });
  if (runId) params.set('runId', runId);
  params.set('modelId', modelId);
  params.set('strategyId', strategyId);
  return `/backtest?${params.toString()}`;
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
}) => {
  const zh = language === 'zh';
  const models = candidatePipeline?.models ?? [];
  const strategyRows = candidatePipeline?.strategyEvaluations ?? [];
  const selected = candidatePipeline?.selectedCombination ?? candidatePipeline?.partialDevelopmentLeader;
  const baselineName = latestTraining?.experimentName || (zh ? 'Qlib 单一选股模型基线' : 'Qlib single stock-selection model baseline');
  const baselineStrategy = zh ? '每日 TopK 等权基线' : 'Daily equal-weight TopK baseline';
  const hasBaselineResult = Boolean(latestTraining?.runId || latestTraining?.experimentName);
  const hasResults = hasBaselineResult || strategyRows.length > 0;

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
                  <td className="max-w-[12rem] truncate px-4 py-3 font-mono text-xs text-secondary-text">{latestTraining?.runId || '-'}</td>
                  <td className="px-4 py-3 text-right">
                    <Link
                      to={backtestHref(latestTraining?.runId, 'latest', 'baseline_daily')}
                      className="btn-secondary h-8 gap-1.5 px-2.5 text-xs"
                      aria-label={`${zh ? '回测' : 'Backtest'} ${baselineName} ${baselineStrategy}`}
                    >
                      <BarChart3 className="h-3.5 w-3.5" />
                      {zh ? '回测' : 'Backtest'}
                    </Link>
                  </td>
                </tr>
              ) : null}
              {strategyRows.map((row) => {
                const model = models.find((item) => item.id === row.modelId);
                const label = strategyLabel(row, language);
                const isSelected = selected?.modelId === row.modelId && selected?.strategyId === row.strategyId;
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
                    <td className="max-w-[12rem] truncate px-4 py-3 font-mono text-xs text-secondary-text">{model?.runId || '-'}</td>
                    <td className="px-4 py-3 text-right">
                      <Link
                        to={backtestHref(model?.runId, row.modelId, row.strategyId)}
                        className="btn-secondary h-8 gap-1.5 px-2.5 text-xs"
                        aria-label={`${zh ? '回测' : 'Backtest'} ${model?.label || row.modelId} ${label}`}
                      >
                        <BarChart3 className="h-3.5 w-3.5" />
                        {zh ? '回测' : 'Backtest'}
                      </Link>
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
