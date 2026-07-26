import type React from 'react';
import { useState, useEffect, useCallback } from 'react';
import { Link } from 'react-router-dom';
import { Check, Minus, X } from 'lucide-react';
import { backtestApi } from '../api/backtest';
import {
  researchApi,
  type ResearchCashControlBacktest,
  type ResearchExperimentRun,
  type ResearchQlibPredictions,
  type ResearchCandidatePipeline,
  type ResearchRobustMetricMap,
  type ResearchRobustSignal,
  type ResearchRobustStrategy,
  type ResearchStatus,
} from '../api/research';
import type { ParsedApiError } from '../api/error';
import { getParsedApiError } from '../api/error';
import { ApiErrorAlert, Card, Badge, EmptyState, Pagination, StatusDot, Tooltip } from '../components/common';
import { useUiLanguage } from '../contexts/UiLanguageContext';
import { formatUiText, type UiLanguage } from '../i18n/uiText';
import {
  BACKTEST_DIRECTION_EXPECTED_LABELS,
  BACKTEST_MOVEMENT_LABELS,
  BACKTEST_OUTCOME_LABELS,
  BACKTEST_PHASE_FILTER_OPTIONS,
  BACKTEST_PHASE_LABELS,
  BACKTEST_STATUS_LABELS,
  BACKTEST_TEXT,
} from '../locales/featureText';
import type {
  BacktestResultItem,
  BacktestRunResponse,
  PerformanceMetrics,
  BacktestPhaseFilter,
} from '../types/backtest';
import { buildDecisionActionLabelMap, getDecisionActionLabel } from '../utils/decisionAction';
import { getMarketPhaseSummaryLabel } from '../utils/marketPhase';
import { resolveMlflowUiUrl } from '../utils/mlflowUiUrl';

const BACKTEST_INPUT_CLASS =
  'input-surface input-focus-glow h-11 w-full rounded-xl border bg-transparent px-4 text-sm transition-all focus:outline-none disabled:cursor-not-allowed disabled:opacity-60';
const BACKTEST_COMPACT_INPUT_CLASS =
  'input-surface input-focus-glow h-10 rounded-xl border bg-transparent px-3 py-2 text-xs transition-all focus:outline-none disabled:cursor-not-allowed disabled:opacity-60';
const DEFAULT_MLFLOW_URL = 'http://127.0.0.1:6006/';
const EXPECTED_QLIB_ARTIFACTS = ['pred.pkl', 'portfolio_analysis', 'task'];
type BacktestText = (typeof BACKTEST_TEXT)[UiLanguage];
type BacktestTab = 'dsa' | 'qlib';

// ============ Helpers ============

function pct(value?: number | null): string {
  if (value == null) return '--';
  return `${value.toFixed(1)}%`;
}

function phaseLabel(row: BacktestResultItem, language: UiLanguage): string {
  const label = getMarketPhaseSummaryLabel(row.marketPhaseSummary, language);
  if (label) {
    return label
      .replace('市场阶段: ', '')
      .replace('市场阶段：', '')
      .replace('Market phase: ', '');
  }
  return (row.marketPhase ? BACKTEST_PHASE_LABELS[language][row.marketPhase] : undefined) || row.marketPhase || '--';
}

function labelFromMap(value: string | null | undefined, labels: Record<string, string>): string {
  if (!value) return '--';
  return labels[value] ?? value;
}

function outcomeBadge(outcome: string | undefined, language: UiLanguage) {
  const labels = BACKTEST_OUTCOME_LABELS[language];
  if (!outcome) return <Badge variant="default">--</Badge>;
  switch (outcome) {
    case 'win':
      return <Badge variant="success" glow>{labels.win}</Badge>;
    case 'loss':
      return <Badge variant="danger" glow>{labels.loss}</Badge>;
    case 'neutral':
      return <Badge variant="warning">{labels.neutral}</Badge>;
    default:
      return <Badge variant="default">{outcome}</Badge>;
  }
}

function statusBadge(status: string, language: UiLanguage) {
  const labels = BACKTEST_STATUS_LABELS[language];
  switch (status) {
    case 'completed':
      return <Badge variant="success">{labels.completed}</Badge>;
    case 'insufficient':
    case 'insufficient_data':
      return <Badge variant="warning">{labels.insufficient}</Badge>;
    case 'error':
      return <Badge variant="danger">{labels.error}</Badge>;
    default:
      return <Badge variant="default">{status}</Badge>;
  }
}

function actualMovementBadge(movement: string | null | undefined, language: UiLanguage) {
  const labels = BACKTEST_MOVEMENT_LABELS[language];
  switch (movement) {
    case 'up':
      return <Badge variant="success">{labels.up}</Badge>;
    case 'down':
      return <Badge variant="danger">{labels.down}</Badge>;
    case 'flat':
      return <Badge variant="warning">{labels.flat}</Badge>;
    default:
      return <Badge variant="default">--</Badge>;
  }
}

function boolIcon(value: boolean | null | undefined, text: BacktestText) {
  if (value === true) {
    return (
      <span
        className="backtest-status-chip backtest-status-chip-success"
        aria-label={text.yes}
      >
        <StatusDot tone="success" className="backtest-status-chip-dot" />
        <Check className="h-3.5 w-3.5" />
      </span>
    );
  }

  if (value === false) {
    return (
      <span
        className="backtest-status-chip backtest-status-chip-danger"
        aria-label={text.no}
      >
        <StatusDot tone="danger" className="backtest-status-chip-dot" />
        <X className="h-3.5 w-3.5" />
      </span>
    );
  }

  return (
    <span
      className="backtest-status-chip backtest-status-chip-neutral"
      aria-label={text.unknown}
    >
      <StatusDot tone="neutral" className="backtest-status-chip-dot" />
      <Minus className="h-3.5 w-3.5" />
    </span>
  );
}

// ============ Metric Row ============

const MetricRow: React.FC<{ label: string; value: string; accent?: boolean }> = ({ label, value, accent }) => (
  <div className="backtest-metric-row">
    <span className="label">{label}</span>
    <span className={`value ${accent ? 'accent' : ''}`}>{value}</span>
  </div>
);

function phaseBreakdownText(metrics: PerformanceMetrics, language: UiLanguage): string | null {
  const breakdown = metrics.diagnostics?.phaseBreakdown;
  if (!breakdown || typeof breakdown !== 'object') return null;
  const item = breakdown as Record<string, unknown>;
  const phaseLabels = BACKTEST_PHASE_LABELS[language];
  const parts = [
    [phaseLabels.premarket, item.premarket],
    [phaseLabels.intraday, item.intraday],
    [phaseLabels.postmarket, item.postmarket],
    [phaseLabels.unknown, item.unknown],
  ]
    .map(([label, value]) => `${label} ${Number(value || 0)}`)
    .join(' / ');
  return parts;
}

function formatQlibMetric(value: number): string {
  if (!Number.isFinite(value)) return '--';
  return Math.abs(value) < 1 ? value.toFixed(4) : value.toFixed(2);
}

function formatRatioPct(value?: number | null): string {
  if (value == null || !Number.isFinite(value)) return '--';
  return `${(value * 100).toFixed(1)}%`;
}

function formatRatioNumber(value?: number | null): string {
  if (value == null || !Number.isFinite(value)) return '--';
  return Math.abs(value) < 10 ? value.toFixed(2) : value.toFixed(1);
}

const CashControlBacktestCard: React.FC<{ report?: ResearchCashControlBacktest | null; text: BacktestText }> = ({ report, text }) => {
  const baseline = report?.comparison?.baseline || {};
  const cashControl = report?.comparison?.cashControl || {};
  const summary = report?.signalSummary || {};
  const rows = [
    {
      label: text.qlibCashControlCumulativeReturn,
      baseline: formatRatioPct(baseline.cumulativeReturn),
      cashControl: formatRatioPct(cashControl.cumulativeReturn),
    },
    {
      label: text.qlibCashControlAnnualizedReturn,
      baseline: formatRatioPct(baseline.annualizedReturn),
      cashControl: formatRatioPct(cashControl.annualizedReturn),
    },
    {
      label: text.qlibCashControlMaxDrawdown,
      baseline: formatRatioPct(baseline.maxDrawdown),
      cashControl: formatRatioPct(cashControl.maxDrawdown),
    },
    {
      label: text.qlibCashControlSharpeRatio,
      baseline: formatRatioNumber(baseline.sharpeRatio ?? baseline.informationRatio),
      cashControl: formatRatioNumber(cashControl.sharpeRatio ?? cashControl.informationRatio),
    },
  ];

  return (
    <Card variant="bordered" padding="md">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <h3 className="text-sm font-semibold text-foreground">{text.qlibCashControlTitle}</h3>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-secondary-text">{text.qlibCashControlDescription}</p>
        </div>
      </div>
      {!report ? (
        <p className="mt-3 text-sm text-secondary-text">{text.qlibCashControlNoReport}</p>
      ) : (
        <>
          <div className="mt-3 grid gap-2 md:grid-cols-3">
            <div className="rounded-xl border border-subtle bg-surface/35 px-3 py-2">
              <p className="text-xs text-muted-text">{text.qlibCashControlLatestRegime}</p>
              <p className="mt-1 font-mono text-sm text-foreground">{summary.latestRegime || '--'}</p>
            </div>
            <div className="rounded-xl border border-subtle bg-surface/35 px-3 py-2">
              <p className="text-xs text-muted-text">{text.qlibCashControlLatestCash}</p>
              <p className="mt-1 font-mono text-sm text-foreground">{formatRatioPct(summary.latestCashRatio)}</p>
            </div>
            <div className="rounded-xl border border-subtle bg-surface/35 px-3 py-2">
              <p className="text-xs text-muted-text">{text.qlibCashControlAverageCash}</p>
              <p className="mt-1 font-mono text-sm text-foreground">{formatRatioPct(summary.averageCashRatio)}</p>
            </div>
          </div>
          <div className="mt-3 overflow-x-auto rounded-xl border border-subtle">
            <table className="min-w-full divide-y divide-subtle text-sm">
              <thead className="bg-surface/40 text-xs text-muted-text">
                <tr>
                  <th scope="col" className="px-3 py-2 text-left font-medium">Metric</th>
                  <th scope="col" className="px-3 py-2 text-right font-medium">{text.qlibCashControlBaseline}</th>
                  <th scope="col" className="px-3 py-2 text-right font-medium">{text.qlibCashControlOverlay}</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-subtle">
                {rows.map((row) => (
                  <tr key={row.label}>
                    <td className="px-3 py-2 text-secondary-text">{row.label}</td>
                    <td className="px-3 py-2 text-right font-mono text-foreground">{row.baseline}</td>
                    <td className="px-3 py-2 text-right font-mono text-foreground">{row.cashControl}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </Card>
  );
};

const QlibPredictionCard: React.FC<{ report?: ResearchQlibPredictions | null; text: BacktestText }> = ({ report, text }) => {
  const copy = text.tabQlib === 'Qlib model/strategy backtest'
    ? {
      title: 'Qlib prediction summary',
      description: 'Read-only model score summary: shows the latest-date highest and lowest ranked instruments for human review, not automated trading.',
      noReport: 'No Qlib prediction summary yet. Run the prediction summary task from the workflow page first.',
      latestDate: 'Latest prediction date',
      rows: 'Latest rows',
      run: 'Run ID',
      top: 'Highest scores',
      bottom: 'Lowest scores',
    }
    : {
      title: 'Qlib 预测摘要',
      description: '只读模型评分摘要：展示最新交易日最高/最低评分股票，用于人工研究和复核，不自动下单。',
      noReport: '暂无 Qlib 预测摘要。先在工作流页面运行“生成 Qlib 预测摘要”。',
      latestDate: '最新预测日',
      rows: '当日样本数',
      run: 'Run ID',
      top: '最高评分',
      bottom: '最低评分',
    };
  const top = (report?.top || []).slice(0, 5);
  const bottom = (report?.bottom || []).slice(0, 5);
  const renderRows = (rows: typeof top) => (
    <div className="mt-2 grid gap-1.5">
      {rows.length === 0 ? (
        <p className="text-xs text-muted-text">--</p>
      ) : rows.map((item) => (
        <div key={`${item.instrument}-${item.score}`} className="flex items-center justify-between gap-3 rounded-lg border border-subtle bg-surface/35 px-2.5 py-1.5 text-xs">
          <span className="min-w-0 truncate font-mono text-foreground">{item.instrument}</span>
          <span className="shrink-0 font-mono text-secondary-text">
            {typeof item.score === 'number' ? formatQlibMetric(item.score) : '--'}
          </span>
        </div>
      ))}
    </div>
  );

  return (
    <Card variant="bordered" padding="md">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <h3 className="text-sm font-semibold text-foreground">{copy.title}</h3>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-secondary-text">{copy.description}</p>
        </div>
        {report?.latestDate ? <Badge variant="info">{report.latestDate}</Badge> : null}
      </div>
      {!report ? (
        <p className="mt-3 text-sm text-secondary-text">{copy.noReport}</p>
      ) : (
        <>
          <div className="mt-3 grid gap-2 md:grid-cols-3">
            <div className="rounded-xl border border-subtle bg-surface/35 px-3 py-2">
              <p className="text-xs text-muted-text">{copy.latestDate}</p>
              <p className="mt-1 font-mono text-sm text-foreground">{report.latestDate || '--'}</p>
            </div>
            <div className="rounded-xl border border-subtle bg-surface/35 px-3 py-2">
              <p className="text-xs text-muted-text">{copy.rows}</p>
              <p className="mt-1 font-mono text-sm text-foreground">{report.summary?.latestRows ?? '--'}</p>
            </div>
            <div className="rounded-xl border border-subtle bg-surface/35 px-3 py-2">
              <p className="text-xs text-muted-text">{copy.run}</p>
              <p className="mt-1 truncate font-mono text-sm text-foreground">{report.runId || '--'}</p>
            </div>
          </div>
          <div className="mt-3 grid gap-3 md:grid-cols-2">
            <div>
              <p className="text-xs font-medium text-muted-text">{copy.top}</p>
              {renderRows(top)}
            </div>
            <div>
              <p className="text-xs font-medium text-muted-text">{copy.bottom}</p>
              {renderRows(bottom)}
            </div>
          </div>
        </>
      )}
    </Card>
  );
};

const CandidatePipelineCard: React.FC<{
  report: ResearchCandidatePipeline;
  text: BacktestText;
}> = ({ report, text }) => {
  const english = text.tabQlib === 'Qlib model/strategy backtest';
  const champion = report.selectedCombination;
  const holdout = champion?.holdout || {};
  const verdict = report.holdoutVerdict?.status;
  const verdictLabel = verdict === 'passed_research_gate'
    ? (english ? 'Passed research gate' : '通过研究门槛')
    : verdict === 'rejected_by_holdout'
      ? (english ? 'Rejected by holdout' : '留出期未通过')
      : (english ? 'No valid final verdict' : '暂无有效终审结论');

  return (
    <Card variant="bordered" padding="md">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-foreground">
            {english ? 'Five-model candidate pipeline' : '五模型候选流程'}
          </h3>
          <p className="mt-1 text-xs text-muted-text">
            {english
              ? 'Models and strategies are selected on 2024-2025; the 2026 holdout is used only for the final check.'
              : '模型和策略只按 2024-2025 选择；2026 留出期只做最终检验，不参与改选。'}
          </p>
        </div>
        <Badge variant={verdict === 'passed_research_gate' ? 'success' : 'default'}>{verdictLabel}</Badge>
      </div>

      <div className="mt-4 grid gap-3 border-y border-subtle py-3 sm:grid-cols-2 lg:grid-cols-4">
        <div>
          <p className="text-xs text-muted-text">{english ? 'Selected models' : '入围模型'}</p>
          <p className="mt-1 text-sm text-foreground">{(report.selectedModels || []).join(', ') || '--'}</p>
        </div>
        <div>
          <p className="text-xs text-muted-text">{english ? 'Selected combination' : '开发期选中组合'}</p>
          <p className="mt-1 text-sm font-semibold text-foreground">
            {champion ? `${champion.modelId} + ${champion.strategyId}` : '--'}
          </p>
        </div>
        <div>
          <p className="text-xs text-muted-text">{english ? 'Latest prediction day' : '最新预测日'}</p>
          <p className="mt-1 font-mono text-sm text-foreground">{report.latestPredictionDay || '--'}</p>
        </div>
        <div>
          <p className="text-xs text-muted-text">{english ? 'Latest evaluable day' : '最新可验证日'}</p>
          <p className="mt-1 font-mono text-sm text-foreground">{report.latestLabelDay || '--'}</p>
        </div>
      </div>

      <div className="mt-3 grid gap-3 sm:grid-cols-3">
        <div>
          <p className="text-xs text-muted-text">{english ? 'Holdout excess return' : '留出期年化超额'}</p>
          <p className="mt-1 font-mono text-base font-semibold text-foreground">
            {formatRatioPct(holdout.annualExcessReturn)}
          </p>
        </div>
        <div>
          <p className="text-xs text-muted-text">{english ? 'Holdout information ratio' : '留出期信息比率'}</p>
          <p className="mt-1 font-mono text-base font-semibold text-foreground">
            {formatRatioNumber(holdout.informationRatio)}
          </p>
        </div>
        <div>
          <p className="text-xs text-muted-text">{english ? 'Holdout max drawdown' : '留出期最大回撤'}</p>
          <p className="mt-1 font-mono text-base font-semibold text-foreground">
            {formatRatioPct(holdout.maxDrawdown)}
          </p>
        </div>
      </div>

      <div className="mt-4 overflow-x-auto">
        <table className="min-w-full text-left text-xs">
          <thead className="text-muted-text">
            <tr>
              <th className="px-2 py-2 font-medium">{english ? 'Model' : '模型'}</th>
              <th className="px-2 py-2 text-right font-medium">{english ? 'Selection IR' : '开发期 IR'}</th>
              <th className="px-2 py-2 text-right font-medium">{english ? 'Holdout IR' : '留出期 IR'}</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-subtle">
            {(report.models || []).map((model) => (
              <tr key={model.id}>
                <td className="px-2 py-2 font-mono text-foreground">{model.id}</td>
                <td className="px-2 py-2 text-right font-mono text-foreground">
                  {formatRatioNumber(model.selection?.informationRatio)}
                </td>
                <td className="px-2 py-2 text-right font-mono text-foreground">
                  {formatRatioNumber(model.holdout?.informationRatio)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-3 text-xs text-muted-text">
        {english
          ? 'Research result only. It does not connect to a broker or place orders automatically.'
          : '仅供研究复核；不连接券商，不自动下单。'}
      </p>
    </Card>
  );
};

function robustMetric(metrics: ResearchRobustMetricMap | undefined, key: string): number | null {
  const value = metrics?.[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

const RobustSignalCard: React.FC<{
  report?: ResearchRobustSignal | null;
  text: BacktestText;
}> = ({ report, text }) => {
  if (!report) return null;
  const english = text.tabQlib === 'Qlib model/strategy backtest';
  const validation = report.validationReports || [];
  const selectedId = report.result?.selectedSignalId;
  const selected = validation.find((item) => item.signalId === selectedId) || validation[0];
  const factor = selected?.factorQuality;
  const processedFactor = factor?.processed;
  const topk = Object.entries(selected?.topkSensitivity?.results || {});
  const stability = selected?.windowStability;
  const states = Object.entries(selected?.marketStateAttribution || {});
  const stateLabels: Record<string, string> = english
    ? { risk_on: 'Risk-on', neutral: 'Neutral', defensive: 'Defensive', risk_off: 'Risk-off' }
    : { risk_on: '风险偏好', neutral: '中性', defensive: '防御', risk_off: '风险关闭' };
  const pctValue = (value: unknown) => (
    typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '--'
  );
  const numberValue = (value: unknown) => (
    typeof value === 'number' && Number.isFinite(value) ? value.toFixed(3) : '--'
  );

  return (
    <Card variant="bordered" padding="md">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <span className="label-uppercase">Goal 1 · Signal</span>
          <h3 className="mt-2 text-sm font-semibold text-foreground">
            {english ? 'Signal stability matrix' : '选股信号稳定性矩阵'}
          </h3>
          <p className="mt-1 text-xs leading-5 text-muted-text">
            {english
              ? 'This is signal research, not a completed trading strategy. 2026 results are locked out.'
              : '这是选股信号研究，不是已经完成的交易策略；2026 结果被锁定，不能参与选型。'}
          </p>
        </div>
        <Badge variant={report.status === 'signal_candidate_ready' ? 'success' : 'warning'}>
          {report.status}
        </Badge>
      </div>

      <div className="mt-4 grid gap-3 sm:grid-cols-4">
        <div className="border-l-2 border-subtle pl-3">
          <p className="text-xs text-muted-text">{english ? 'Run ID' : 'Run ID'}</p>
          <p className="mt-1 truncate font-mono text-sm text-foreground">{report.runId || '--'}</p>
        </div>
        <div className="border-l-2 border-subtle pl-3">
          <p className="text-xs text-muted-text">{english ? 'Selected signal' : '入选信号'}</p>
          <p className="mt-1 truncate font-mono text-sm text-foreground">{selectedId || '--'}</p>
        </div>
        <div className="border-l-2 border-subtle pl-3">
          <p className="text-xs text-muted-text">{english ? 'Model' : '模型'}</p>
          <p className="mt-1 truncate font-mono text-sm text-foreground">{report.result?.selectedModelId || '--'}</p>
        </div>
        <div className="border-l-2 border-subtle pl-3">
          <p className="text-xs text-muted-text">{english ? 'Next gate' : '下一门槛'}</p>
          <p className="mt-1 break-words font-mono text-sm text-foreground">{report.nextGate || '--'}</p>
        </div>
      </div>

      <div className="mt-5 grid gap-3 lg:grid-cols-2">
        <div>
          <h4 className="text-xs font-semibold text-foreground">{english ? 'Factor quality' : '因子质量'}</h4>
          <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
            <div className="rounded-lg border border-subtle px-3 py-2">
              <span className="text-muted-text">{english ? 'Raw feature count' : '原始因子数'}</span>
              <p className="mt-1 font-mono text-foreground">{factor?.featureCountMin ?? '--'} - {factor?.featureCountMax ?? '--'}</p>
            </div>
            <div className="rounded-lg border border-subtle px-3 py-2">
              <span className="text-muted-text">{english ? 'Raw max missing' : '原始最高缺失率'}</span>
              <p className="mt-1 font-mono text-foreground">{pctValue(factor?.maxMissingRate)}</p>
            </div>
            <div className="rounded-lg border border-subtle px-3 py-2">
              <span className="text-muted-text">{english ? 'Model input count' : '模型输入因子数'}</span>
              <p className="mt-1 font-mono text-foreground">{processedFactor?.featureCountMin ?? '--'} - {processedFactor?.featureCountMax ?? '--'}</p>
            </div>
            <div className="rounded-lg border border-subtle px-3 py-2">
              <span className="text-muted-text">{english ? 'Input max missing' : '模型输入最高缺失率'}</span>
              <p className="mt-1 font-mono text-foreground">{pctValue(processedFactor?.maxMissingRate)}</p>
            </div>
            <div className="rounded-lg border border-subtle px-3 py-2">
              <span className="text-muted-text">{english ? 'High-correlation pairs' : '高相关因子对'}</span>
              <p className="mt-1 font-mono text-foreground">{factor?.highCorrelationPairCountMax ?? '--'}</p>
            </div>
            <div className="rounded-lg border border-subtle px-3 py-2">
              <span className="text-muted-text">{english ? 'Contract hash' : '契约哈希'}</span>
              <p className="mt-1 truncate font-mono text-foreground">{report.factorContract?.contractHash || '--'}</p>
            </div>
          </div>
        </div>
        <div>
          <h4 className="text-xs font-semibold text-foreground">{english ? 'Window stability' : '训练窗口稳定性'}</h4>
          <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
            <div className="rounded-lg border border-subtle px-3 py-2">
              <span className="text-muted-text">{english ? 'Rank correlation' : '排名相关性'}</span>
              <p className="mt-1 font-mono text-foreground">{numberValue(stability?.rankCorrelationMedian)}</p>
            </div>
            <div className="rounded-lg border border-subtle px-3 py-2">
              <span className="text-muted-text">Top50 overlap</span>
              <p className="mt-1 font-mono text-foreground">{pctValue(stability?.topkOverlapMean)}</p>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-5 overflow-x-auto">
        <h4 className="text-xs font-semibold text-foreground">{english ? 'Label scheme comparison' : '标签方案对比'}</h4>
        <table className="mt-2 min-w-full text-left text-xs">
          <thead className="text-muted-text"><tr>
            <th className="px-2 py-2 font-medium">{english ? 'Signal' : '信号方案'}</th>
            <th className="px-2 py-2 font-medium">{english ? 'Model' : '模型'}</th>
            <th className="px-2 py-2 text-right font-medium">{english ? 'Median RankIC' : 'RankIC 中位数'}</th>
            <th className="px-2 py-2 text-right font-medium">{english ? 'Annual excess' : '年化超额'}</th>
            <th className="px-2 py-2 text-right font-medium">{english ? 'Gate' : '门槛'}</th>
          </tr></thead>
          <tbody className="divide-y divide-subtle">
            {validation.map((item) => (
              <tr key={`${item.signalId}-${item.modelId}`}>
                <td className="px-2 py-2 font-mono text-foreground">{item.signalId || '--'}</td>
                <td className="px-2 py-2 font-mono text-foreground">{item.modelId || '--'}</td>
                <td className="px-2 py-2 text-right font-mono text-foreground">{numberValue(item.values?.medianQuarterRankIc)}</td>
                <td className="px-2 py-2 text-right font-mono text-foreground">{pctValue(item.values?.annualExcessReturn)}</td>
                <td className="px-2 py-2 text-right">{item.passed ? (english ? 'Pass' : '通过') : (english ? 'Fail' : '未通过')}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mt-5 grid gap-4 lg:grid-cols-2">
        <div className="overflow-x-auto">
          <h4 className="text-xs font-semibold text-foreground">TopK {english ? 'sensitivity' : '敏感性'}</h4>
          <table className="mt-2 min-w-full text-left text-xs"><tbody className="divide-y divide-subtle">
            {topk.map(([key, value]) => <tr key={key}><td className="px-2 py-2 font-mono text-foreground">Top{key}</td><td className="px-2 py-2 text-right font-mono text-foreground">{pctValue(value.meanLabel)}</td></tr>)}
          </tbody></table>
        </div>
        <div className="overflow-x-auto">
          <h4 className="text-xs font-semibold text-foreground">{english ? 'Four market states' : '四类市场状态'}</h4>
          <table className="mt-2 min-w-full text-left text-xs"><tbody className="divide-y divide-subtle">
            {states.map(([key, value]) => <tr key={key}><td className="px-2 py-2 text-foreground">{stateLabels[key] || key}</td><td className="px-2 py-2 text-right font-mono text-foreground">{pctValue(value.annualExcessReturn)}</td><td className="px-2 py-2 text-right font-mono text-muted-text">{value.periodCount ?? 0} {english ? 'periods' : '期'}</td></tr>)}
          </tbody></table>
        </div>
      </div>
    </Card>
  );
};

const RobustStrategyCard: React.FC<{
  report?: ResearchRobustStrategy | null;
  text: BacktestText;
}> = ({ report, text }) => {
  if (!report) return null;
  const english = text.tabQlib === 'Qlib model/strategy backtest';
  const statusLabels: Record<string, string> = english
    ? {
        missing: 'Not started',
        blocked_data: 'Blocked by data',
        development_failed: 'Development failed',
        research_candidate_ready: 'Research candidate frozen',
        shadow_observation: '60-day shadow tracking',
        robust_strategy_qualified: 'Robust strategy qualified',
      }
    : {
        missing: '未启动',
        blocked_data: '数据阻断',
        development_failed: '开发未通过',
        research_candidate_ready: '研究候选已冻结',
        shadow_observation: '60日纸面跟踪中',
        robust_strategy_qualified: '鲁棒策略合格',
      };
  const statusLabel = statusLabels[report.status] || report.status;
  const statusVariant = report.status === 'robust_strategy_qualified'
    ? 'success'
    : report.status === 'development_failed'
      ? 'danger'
      : report.status === 'blocked_data'
        ? 'warning'
        : 'info';
  const development = report.development;
  const strategies = development?.strategyMatrix || [];
  const selectedStrategy = strategies.find((item) => item.strategyId === development?.selectedStrategy)
    || null;
  const quarters = selectedStrategy?.quarterMetrics || [];
  const regimes = Object.entries(selectedStrategy?.regimeMetrics || {});
  const boardNames: Record<string, string> = english
    ? { main: 'Main board', growth: 'ChiNext', star: 'STAR Market' }
    : { main: '主板', growth: '创业板', star: '科创板' };
  const boards = ['main', 'growth', 'star']
    .map((name) => [name, selectedStrategy?.boardAttribution?.[name]] as const)
    .filter((entry): entry is readonly [string, ResearchRobustMetricMap] => Boolean(entry[1]));
  const acceptanceValues = selectedStrategy?.acceptance?.values;
  const acceptanceThresholds = selectedStrategy?.acceptance?.thresholds;
  const drawdownValue = Math.max(
    robustMetric(acceptanceValues, 'worstMaxDrawdownAbs') ?? 0,
    robustMetric(acceptanceValues, 'aggregateMaxDrawdownAbs') ?? 0,
  );
  const acceptanceRows = [
    {
      label: english ? 'Positive excess quarters' : '正超额季度比例',
      value: robustMetric(acceptanceValues, 'positiveExcessQuarterRatio'),
      threshold: robustMetric(acceptanceThresholds, 'positiveExcessQuarterRatioMin'),
      kind: 'pct',
      relation: '≥',
    },
    {
      label: english ? 'Annual excess return' : '年化超额收益',
      value: robustMetric(acceptanceValues, 'aggregateAnnualExcessReturn'),
      threshold: robustMetric(acceptanceThresholds, 'annualExcessReturnMin'),
      kind: 'pct',
      relation: '≥',
    },
    {
      label: english ? 'Median quarterly IR' : '季度 IR 中位数',
      value: robustMetric(acceptanceValues, 'medianQuarterInformationRatio'),
      threshold: robustMetric(acceptanceThresholds, 'medianQuarterInformationRatioMin'),
      kind: 'number',
      relation: '≥',
    },
    {
      label: english ? 'Positive RankIC intervals' : 'RankIC 为正区间比例',
      value: robustMetric(acceptanceValues, 'positiveRankIcIntervalRatio'),
      threshold: robustMetric(acceptanceThresholds, 'positiveRankIcIntervalRatioMin'),
      kind: 'pct',
      relation: '≥',
    },
    {
      label: english ? 'Window rank correlation' : '窗口排名相关性',
      value: robustMetric(acceptanceValues, 'windowRankCorrelationMedian'),
      threshold: robustMetric(acceptanceThresholds, 'windowRankCorrelationMedianMin'),
      kind: 'number',
      relation: '≥',
    },
    {
      label: 'Top50 overlap',
      value: robustMetric(acceptanceValues, 'top50OverlapMean'),
      threshold: robustMetric(acceptanceThresholds, 'top50OverlapMeanMin'),
      kind: 'pct',
      relation: '≥',
    },
    {
      label: english ? 'Maximum drawdown' : '最大回撤',
      value: drawdownValue || null,
      threshold: robustMetric(acceptanceThresholds, 'developmentMaxDrawdownAbsMax'),
      kind: 'pct',
      relation: '≤',
    },
    {
      label: english ? '40bp stress excess' : '40bp 压力超额',
      value: robustMetric(acceptanceValues, 'stressExcessReturn'),
      threshold: robustMetric(acceptanceThresholds, 'stressCostExcessReturnMinExclusive'),
      kind: 'pct',
      relation: '>',
    },
    {
      label: english ? 'Delayed execution excess' : '延迟执行超额',
      value: robustMetric(acceptanceValues, 'delayedExcessReturn'),
      threshold: robustMetric(acceptanceThresholds, 'delayedExecutionExcessReturnMinExclusive'),
      kind: 'pct',
      relation: '>',
    },
  ].filter((row) => row.value !== null && row.threshold !== null);
  const formatAcceptanceMetric = (value: number | null, kind: string) => (
    kind === 'pct' ? formatRatioPct(value) : formatRatioNumber(value)
  );
  const marketTone = (value: number | null) => (
    value == null || value === 0 ? 'text-foreground' : value > 0 ? 'text-danger' : 'text-success'
  );
  const shadowDays = robustMetric(report.shadow?.metrics, 'tradingDays') ?? 0;
  const remainingDays = report.shadow?.verdict?.remainingDays;
  const shadowDailyFailed = report.shadowDaily?.status === 'development_failed';

  return (
    <Card variant="bordered" padding="md">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <span className="label-uppercase">Robust CSI800</span>
          <h3 className="mt-2 text-sm font-semibold text-foreground">
            {english ? 'Robustness matrix' : '鲁棒策略矩阵'}
          </h3>
          <p className="mt-1 max-w-3xl text-xs leading-5 text-muted-text">
            {english
              ? 'Development selection, one-time 2026 confirmation, and future shadow observations are separate gates.'
              : '开发期选型、2026 单次确认和未来纸面跟踪是三个独立门槛；前两项通过也不等于最终合格。'}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant={statusVariant}>{statusLabel}</Badge>
          <a href="/experiments/#/experiments" className="btn-secondary h-8 px-2.5 text-xs">
            {english ? 'View robust experiments' : '查看鲁棒实验'}
          </a>
        </div>
      </div>

      {report.reasonCode ? (
        <p className="mt-3 break-words font-mono text-xs text-warning">{report.reasonCode}</p>
      ) : null}

      {report.shadowDaily && report.shadowDaily.status !== 'missing' ? (
        <div className={`mt-3 border-l-2 px-3 py-2 ${shadowDailyFailed ? 'border-danger bg-danger/5' : 'border-cyan bg-cyan/5'}`}>
          <p className="text-xs font-semibold text-foreground">
            {shadowDailyFailed
              ? (english ? 'Latest shadow task failed' : '最近纸面任务失败')
              : (english ? 'Latest shadow task recorded' : '最近纸面任务已记录')}
          </p>
          <p className="mt-1 break-words font-mono text-xs text-secondary-text">
            {report.shadowDaily.asOf || '--'}
            {report.shadowDaily.reasonCode ? ` · ${report.shadowDaily.reasonCode}` : ''}
          </p>
        </div>
      ) : null}

      <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <div className="border-l-2 border-subtle pl-3">
          <p className="text-xs text-muted-text">{english ? 'Selected models' : '入围模型'}</p>
          <p className="mt-1 text-sm text-foreground">{(development?.selectedModels || []).join(', ') || '--'}</p>
        </div>
        <div className="border-l-2 border-subtle pl-3">
          <p className="text-xs text-muted-text">{english ? 'Frozen strategy' : '冻结策略'}</p>
          <p className="mt-1 font-mono text-sm text-foreground">{development?.selectedStrategy || '--'}</p>
        </div>
        <div className="border-l-2 border-subtle pl-3">
          <p className="text-xs text-muted-text">{english ? 'Confirmation' : '确认期'}</p>
          <p className="mt-1 text-sm text-foreground">{report.confirmation?.verdict?.passed ? (english ? 'Passed' : '通过') : (english ? 'Not passed' : '未通过')}</p>
        </div>
        <div className="border-l-2 border-subtle pl-3">
          <p className="text-xs text-muted-text">
            {english ? `Shadow tracking ${shadowDays} / 60 days` : `纸面跟踪 ${shadowDays} / 60 日`}
          </p>
          <p className="mt-1 text-sm text-foreground">
            {remainingDays == null ? '--' : (english ? `${remainingDays} days remaining` : `还差 ${remainingDays} 日`)}
          </p>
        </div>
      </div>

      <div className="mt-5">
        <h4 className="text-xs font-semibold text-foreground">{english ? 'Model stability' : '模型稳定性'}</h4>
        <div className="mt-2 overflow-x-auto">
          <table className="min-w-full text-left text-xs">
            <thead className="text-muted-text">
              <tr>
                <th className="px-2 py-2 font-medium">{english ? 'Model' : '模型'}</th>
                <th className="px-2 py-2 text-right font-medium">{english ? 'Eligible' : '入围'}</th>
                <th className="px-2 py-2 text-right font-medium">{english ? 'Window rank corr.' : '窗口排名相关'}</th>
                <th className="px-2 py-2 text-right font-medium">Top50 overlap</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-subtle">
              {(development?.modelStability || []).map((model, index) => (
                <tr key={model.modelId || index}>
                  <td className="px-2 py-2 font-mono text-foreground">{model.modelId || '--'}</td>
                  <td className="px-2 py-2 text-right text-foreground">{model.eligible ? (english ? 'Yes' : '是') : (english ? 'No' : '否')}</td>
                  <td className="px-2 py-2 text-right font-mono text-foreground">{formatRatioNumber(model.stability?.rankCorrelationMedian)}</td>
                  <td className="px-2 py-2 text-right font-mono text-foreground">{formatRatioPct(model.stability?.topkOverlapMean)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="mt-5">
        <h4 className="text-xs font-semibold text-foreground">{english ? 'Strategy cost and delay stress tests' : '策略成本与延迟压力测试'}</h4>
        <div className="mt-2 overflow-x-auto">
          <table className="min-w-full text-left text-xs">
            <thead className="text-muted-text">
              <tr>
                <th className="px-2 py-2 font-medium">{english ? 'Strategy' : '策略'}</th>
                <th className="px-2 py-2 font-medium">{english ? 'Role' : '角色'}</th>
                <th className="px-2 py-2 text-right font-medium">{english ? 'Base excess' : '基准成本超额'}</th>
                <th className="px-2 py-2 text-right font-medium">{english ? '40bp stress' : '40bp 压力'}</th>
                <th className="px-2 py-2 text-right font-medium">{english ? 'One-day delay' : '延迟一日'}</th>
                <th className="px-2 py-2 text-right font-medium">{english ? 'All gates' : '全部门槛'}</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-subtle">
              {strategies.map((strategy, index) => (
                <tr key={strategy.strategyId || index}>
                  <td className="px-2 py-2 font-mono text-foreground">{strategy.strategyId || '--'}</td>
                  <td className="px-2 py-2 text-foreground">
                    {strategy.eligibleForFinalSelection
                      ? (english ? 'Final candidate' : '可冻结')
                      : (english ? 'Ablation' : '对照')}
                  </td>
                  <td className="px-2 py-2 text-right font-mono text-foreground">{formatRatioPct(robustMetric(strategy.baseline?.metrics, 'annualExcessReturn'))}</td>
                  <td className="px-2 py-2 text-right font-mono text-foreground">{formatRatioPct(robustMetric(strategy.stress?.metrics, 'annualExcessReturn'))}</td>
                  <td className="px-2 py-2 text-right font-mono text-foreground">{formatRatioPct(robustMetric(strategy.delayed?.metrics, 'annualExcessReturn'))}</td>
                  <td className="px-2 py-2 text-right text-foreground">{strategy.acceptance?.passed ? (english ? 'Pass' : '通过') : (english ? 'Fail' : '未通过')}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {acceptanceRows.length > 0 ? (
        <div className="mt-5">
          <h4 className="text-xs font-semibold text-foreground">{english ? 'Acceptance values and thresholds' : '验收数值与门槛'}</h4>
          <div className="mt-2 overflow-x-auto">
            <table className="min-w-full text-left text-xs">
              <thead className="text-muted-text">
                <tr>
                  <th className="px-2 py-2 font-medium">{english ? 'Gate' : '指标'}</th>
                  <th className="px-2 py-2 text-right font-medium">{english ? 'Actual' : '实际值'}</th>
                  <th className="px-2 py-2 text-right font-medium">{english ? 'Threshold' : '门槛'}</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-subtle">
                {acceptanceRows.map((row) => (
                  <tr key={row.label}>
                    <td className="px-2 py-2 text-foreground">{row.label}</td>
                    <td className="px-2 py-2 text-right font-mono text-foreground">{formatAcceptanceMetric(row.value, row.kind)}</td>
                    <td className="px-2 py-2 text-right font-mono text-muted-text">{row.relation} {formatAcceptanceMetric(row.threshold, row.kind)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : null}

      {quarters.length > 0 ? (
        <div className="mt-5">
          <h4 className="text-xs font-semibold text-foreground">{english ? 'Quarterly out-of-sample results' : '季度样本外表现'}</h4>
          <div className="mt-2 overflow-x-auto">
            <table className="min-w-full text-left text-xs">
              <thead className="text-muted-text">
                <tr>
                  <th className="px-2 py-2 font-medium">{english ? 'Quarter' : '季度'}</th>
                  <th className="px-2 py-2 text-right font-medium">{english ? 'Excess return' : '超额收益'}</th>
                  <th className="px-2 py-2 text-right font-medium">RankIC</th>
                  <th className="px-2 py-2 text-right font-medium">{english ? 'Max drawdown' : '最大回撤'}</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-subtle">
                {quarters.map((quarter, index) => (
                  <tr key={String(quarter.quarter || index)}>
                    <td className="px-2 py-2 font-mono text-foreground">{String(quarter.quarter || '--')}</td>
                    <td className="px-2 py-2 text-right font-mono text-foreground">{formatRatioPct(robustMetric(quarter, 'annualExcessReturn'))}</td>
                    <td className="px-2 py-2 text-right font-mono text-foreground">{formatRatioNumber(robustMetric(quarter, 'rankIc'))}</td>
                    <td className="px-2 py-2 text-right font-mono text-foreground">{formatRatioPct(robustMetric(quarter, 'maximumDrawdown'))}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : null}

      {regimes.length > 0 ? (
        <div className="mt-5">
          <h4 className="text-xs font-semibold text-foreground">{english ? 'Market regimes' : '牛熊与市场状态'}</h4>
          <div className="mt-2 grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
            {regimes.map(([name, metrics]) => (
              <div key={name} className="border-l-2 border-subtle pl-3">
                <p className="font-mono text-xs text-muted-text">{name}</p>
                <p className="mt-1 font-mono text-sm text-foreground">{formatRatioPct(robustMetric(metrics, 'annualExcessReturn'))}</p>
                <p className="mt-1 text-xs text-muted-text">{String(metrics.days ?? '--')} {english ? 'days' : '日'}</p>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {boards.length > 0 ? (
        <div className="mt-5">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h4 className="text-xs font-semibold text-foreground">{english ? 'Board and cash attribution' : '板块与现金归因'}</h4>
            <Badge
              variant={selectedStrategy?.attributionReconciliation?.reconciled ? 'success' : 'danger'}
              size="sm"
              className="max-w-full whitespace-normal break-words text-center leading-4 shadow-none"
            >
              {selectedStrategy?.attributionReconciliation?.reconciled
                ? (english ? 'Compounded reconciliation passed' : '复利对账通过')
                : (english ? 'Reconciliation failed' : '对账失败')}
            </Badge>
          </div>
          <div className="mt-2 overflow-x-auto">
            <table className="min-w-full text-left text-xs">
              <thead className="text-muted-text">
                <tr>
                  <th className="px-2 py-2 font-medium">{english ? 'Board' : '板块'}</th>
                  <th className="px-2 py-2 text-right font-medium">{english ? 'Avg. exposure' : '平均仓位'}</th>
                  <th className="px-2 py-2 text-right font-medium">{english ? 'Peak exposure' : '峰值仓位'}</th>
                  <th className="px-2 py-2 text-right font-medium">{english ? 'Return contribution' : '收益贡献'}</th>
                  <th className="px-2 py-2 text-right font-medium">{english ? 'Excess contribution' : '超额贡献'}</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-subtle">
                {boards.map(([name, metrics]) => {
                  const returnContribution = robustMetric(metrics, 'returnContribution');
                  const excessContribution = robustMetric(metrics, 'excessContribution');
                  return (
                    <tr key={name}>
                      <td className="px-2 py-2 text-foreground">{boardNames[name] || name}</td>
                      <td className="px-2 py-2 text-right font-mono text-foreground">{formatRatioPct(robustMetric(metrics, 'averageExposure'))}</td>
                      <td className="px-2 py-2 text-right font-mono text-foreground">{formatRatioPct(robustMetric(metrics, 'maxExposure'))}</td>
                      <td className={`px-2 py-2 text-right font-mono ${marketTone(returnContribution)}`}>{formatRatioPct(returnContribution)}</td>
                      <td className={`px-2 py-2 text-right font-mono ${marketTone(excessContribution)}`}>{formatRatioPct(excessContribution)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className={`mt-2 text-xs ${marketTone(robustMetric(selectedStrategy?.attributionReconciliation, 'cashExcessContribution'))}`}>
            {english ? 'Cash excess contribution' : '现金超额贡献'} {formatRatioPct(robustMetric(selectedStrategy?.attributionReconciliation, 'cashExcessContribution'))}
          </p>
        </div>
      ) : null}

      <div className="mt-5 grid gap-3 sm:grid-cols-2">
        <div className="border-t border-subtle pt-3">
          <p className="text-xs font-semibold text-foreground">{english ? 'Confirmation period' : '确认期'}</p>
          <p className="mt-2 text-xs text-muted-text">{report.confirmation?.latestMarketDay || '--'}</p>
          <p className="mt-1 font-mono text-sm text-foreground">
            {english ? 'Excess ' : '年化超额 '}{formatRatioPct(robustMetric(report.confirmation?.metrics, 'annualExcessReturn'))}
            {' · IR '}{formatRatioNumber(robustMetric(report.confirmation?.metrics, 'informationRatio'))}
          </p>
        </div>
        <div className="border-t border-subtle pt-3">
          <p className="text-xs font-semibold text-foreground">{english ? 'Shadow observation' : '纸面跟踪'}</p>
          <p className="mt-2 font-mono text-sm text-foreground">
            {english ? 'Net excess ' : '累计净超额 '}{formatRatioPct(robustMetric(report.shadow?.metrics, 'netExcessReturn'))}
          </p>
          <p className="mt-1 text-xs text-muted-text">
            {english ? 'Task success ' : '任务成功率 '}{formatRatioPct(robustMetric(report.shadow?.metrics, 'taskSuccessRatio'))}
          </p>
        </div>
      </div>

      <p className="mt-4 text-xs text-muted-text">
        {english
          ? `MLflow experiments: ${(report.mlflowExperiments || []).join(', ') || '--'}. Research only; no broker connection or automatic orders.`
          : `MLflow 实验：${(report.mlflowExperiments || []).join(', ') || '--'}。仅供研究复核；不连接券商，不自动下单。`}
      </p>
    </Card>
  );
};

// ============ Performance Card ============

const PerformanceCard: React.FC<{ metrics: PerformanceMetrics; title: string; language: UiLanguage }> = ({ metrics, title, language }) => {
  const text = BACKTEST_TEXT[language];
  const phaseText = phaseBreakdownText(metrics, language);
  return (
    <Card variant="gradient" padding="md" className="animate-fade-in">
      <div className="mb-3">
        <span className="label-uppercase">{title}</span>
      </div>
      <MetricRow label={text.directionAccuracy} value={pct(metrics.directionAccuracyPct)} accent />
      <MetricRow label={text.winRate} value={pct(metrics.winRatePct)} accent />
      <MetricRow label={text.avgSimulatedReturn} value={pct(metrics.avgSimulatedReturnPct)} />
      <MetricRow label={text.avgStockReturn} value={pct(metrics.avgStockReturnPct)} />
      <MetricRow label={text.stopLossTriggerRate} value={pct(metrics.stopLossTriggerRate)} />
      <MetricRow label={text.takeProfitTriggerRate} value={pct(metrics.takeProfitTriggerRate)} />
      <MetricRow label={text.avgDaysToFirstHit} value={metrics.avgDaysToFirstHit != null ? metrics.avgDaysToFirstHit.toFixed(1) : '--'} />
      <div className="backtest-metric-footer">
        <span className="text-xs text-muted-text">{text.evaluationCount}</span>
        <span className="text-xs text-secondary-text font-mono">
          {Number(metrics.completedCount)} / {Number(metrics.totalEvaluations)}
        </span>
      </div>
      <div className="flex items-center justify-between">
        <span className="text-xs text-muted-text">{text.outcomeSummary}</span>
        <span className="text-xs font-mono">
          <span className="text-success">{metrics.winCount}</span>
          {' / '}
          <span className="text-danger">{metrics.lossCount}</span>
          {' / '}
          <span className="text-warning">{metrics.neutralCount}</span>
        </span>
      </div>
      {phaseText ? (
        <div className="mt-3 border-t border-white/10 pt-2 text-xs text-muted-text">
          {formatUiText(text.phaseDistribution, { text: phaseText })}
        </div>
      ) : null}
    </Card>
  );
};

// ============ Run Summary ============

const RunSummary: React.FC<{ data: BacktestRunResponse; language: UiLanguage }> = ({ data, language }) => {
  const text = BACKTEST_TEXT[language];
  return (
  <div className="backtest-summary animate-fade-in">
    <span className="label">{text.processed} <span className="value">{data.processed}</span></span>
    <span className="label">{text.saved} <span className="value primary">{data.saved}</span></span>
    <span className="label">{text.completed} <span className="value success">{data.completed}</span></span>
    <span className="label">{text.insufficient} <span className="value warning">{data.insufficient}</span></span>
    {data.errors > 0 && (
      <span className="label">{text.errors} <span className="value danger">{data.errors}</span></span>
    )}
  </div>
  );
};

const QlibRunCard: React.FC<{ run: ResearchExperimentRun; text: BacktestText }> = ({ run, text }) => {
  const metrics = Object.entries(run.metrics || {}).slice(0, 6);
  return (
    <Card variant="bordered" padding="md" className="min-h-[12rem]">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="min-w-0">
          <p className="truncate text-sm font-semibold text-foreground">{run.experimentName || text.qlibExperimentFallback}</p>
          <p className="mt-1 truncate font-mono text-xs text-muted-text">{run.runId || '--'}</p>
        </div>
        <Badge variant={run.status === 'completed' ? 'success' : 'default'} size="sm">{run.status}</Badge>
      </div>
      <div className="mt-3 grid grid-cols-2 gap-2 text-sm">
        <div className="rounded-xl border border-subtle bg-surface/35 px-3 py-2">
          <p className="text-xs text-muted-text">{text.qlibModel}</p>
          <p className="mt-1 truncate text-foreground">{run.model || '--'}</p>
        </div>
        <div className="rounded-xl border border-subtle bg-surface/35 px-3 py-2">
          <p className="text-xs text-muted-text">{text.qlibStrategy}</p>
          <p className="mt-1 truncate text-foreground">{run.strategy || '--'}</p>
        </div>
      </div>
      {run.datasetRange ? (
        <p className="mt-3 text-xs text-secondary-text">{formatUiText(text.qlibDatasetRange, { range: run.datasetRange })}</p>
      ) : null}
      {run.conclusion ? (
        <p className="mt-2 text-sm leading-5 text-secondary-text">{run.conclusion}</p>
      ) : null}
      {metrics.length > 0 ? (
        <div className="mt-3 grid gap-2">
          {metrics.map(([name, value]) => (
            <div key={name} className="flex items-center justify-between gap-3 text-xs">
              <span className="min-w-0 truncate text-muted-text">{name}</span>
              <span className="shrink-0 font-mono text-secondary-text">
                {typeof value === 'number' ? formatQlibMetric(value) : String(value)}
              </span>
            </div>
          ))}
        </div>
      ) : null}
    </Card>
  );
};

const QlibArtifactsCard: React.FC<{ status: ResearchStatus | null; text: BacktestText }> = ({ status, text }) => {
  const latestTraining = status?.latestTraining;
  const artifacts = (latestTraining?.artifacts || [])
    .filter((item): item is string => typeof item === 'string' && item.trim().length > 0)
    .slice(0, 12);
  const visibleArtifacts = artifacts.length > 0 ? artifacts : EXPECTED_QLIB_ARTIFACTS;
  const artifactLabel = text.qlibArtifactFiles;
  const candidateRows: Array<{ label: string; value?: string | null }> = [
    { label: text.qlibArtifactProvider, value: latestTraining?.provider },
    {
      label: text.qlibArtifactProviderPath,
      value: latestTraining?.providerUri || latestTraining?.qlibProvider || status?.qlibProvider?.path,
    },
    { label: text.qlibArtifactSourceConfig, value: latestTraining?.sourceConfig },
    { label: text.qlibArtifactRuntimeConfig, value: latestTraining?.runtimeConfig },
    { label: text.qlibArtifactMlflowRun, value: latestTraining?.runId },
  ];
  const rows = candidateRows.filter((row): row is { label: string; value: string } => Boolean(row.value));
  const hasDetails = rows.length > 0 || artifacts.length > 0;

  return (
    <Card variant="bordered" padding="md">
      <div className="flex flex-col gap-2">
        <h3 className="text-sm font-semibold text-foreground">{text.qlibArtifactsTitle}</h3>
        <p className="max-w-3xl text-sm leading-6 text-secondary-text">{text.qlibArtifactsDescription}</p>
      </div>
      {!hasDetails ? (
        <p className="mt-3 text-sm text-secondary-text">{text.qlibArtifactNoData}</p>
      ) : null}
      <div className="mt-3 grid gap-3">
        {hasDetails ? (
          <>
          {rows.map((row) => (
            <div key={row.label} className="rounded-xl border border-subtle bg-surface/35 px-3 py-2">
              <p className="text-xs text-muted-text">{row.label}</p>
              <p className="mt-1 break-all font-mono text-xs leading-5 text-foreground">{row.value}</p>
            </div>
          ))}
          </>
        ) : null}
        <div className="rounded-xl border border-subtle bg-surface/35 px-3 py-2">
          <p className="text-xs text-muted-text">{artifactLabel}</p>
          <div className="mt-2 flex flex-wrap gap-2">
            {visibleArtifacts.map((artifact) => (
              <Badge key={artifact} variant={artifacts.length > 0 ? 'history' : 'default'} size="sm" className="rounded-lg font-mono">
                {artifact}
              </Badge>
            ))}
          </div>
        </div>
      </div>
    </Card>
  );
};

const QlibBacktestPanel: React.FC<{
  status: ResearchStatus | null;
  loading: boolean;
  error: string | null;
  text: BacktestText;
}> = ({ status, loading, error, text }) => {
  const latestTraining = status?.latestTraining;
  const cashControlBacktest = status?.cashControlBacktest;
  const qlibPredictions = status?.qlibPredictions;
  const candidatePipeline = status?.candidatePipeline;
  const runs = status?.experimentNotebook?.runs || [];
  const metrics = Object.entries(latestTraining?.metrics || {}).slice(0, 8);
  const mlflowUrl = resolveMlflowUiUrl(status?.mlflow?.url || DEFAULT_MLFLOW_URL);

  return (
    <main className="min-h-0 flex-1 overflow-y-auto p-3">
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_360px]">
        <section className="flex flex-col gap-4">
          <Card variant="bordered" padding="md">
            <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
              <div>
                <span className="label-uppercase">{text.qlibEyebrow}</span>
                <h2 className="mt-2 text-xl font-semibold text-foreground">{text.qlibTitle}</h2>
                <p className="mt-2 max-w-2xl text-sm leading-6 text-secondary-text">
                  {text.qlibDescription}
                </p>
              </div>
              <div className="flex flex-wrap gap-2">
                <a href={mlflowUrl} target="_blank" rel="noreferrer" className="btn-secondary h-9 px-3 text-sm">
                  {text.qlibOpenMlflow}
                </a>
                <Link to="/workflow" className="btn-primary h-9 px-3 text-sm">
                  {text.qlibRunTrainingTask}
                </Link>
              </div>
            </div>
            {error ? (
              <ApiErrorAlert
                error={{
                  title: text.qlibStatusReadFailed,
                  message: error,
                  rawMessage: error,
                  category: 'unknown',
                }}
                className="mt-4"
              />
            ) : null}
            {loading ? (
              <p className="mt-4 text-sm text-secondary-text">{text.qlibLoading}</p>
            ) : null}
          </Card>

          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
            <Card variant="bordered" padding="md">
              <span className="label-uppercase">Provider</span>
              <p className="mt-3 text-lg font-semibold text-foreground">{status?.qlibProvider?.calendarRange || '--'}</p>
              <p className="mt-2 text-xs text-muted-text">{text.qlibProviderInstruments} {status?.qlibProvider?.instruments ?? '--'} · {text.qlibProviderCalendarDays} {status?.qlibProvider?.calendarDays ?? '--'}</p>
            </Card>
            <Card variant="bordered" padding="md">
              <span className="label-uppercase">Experiment</span>
              <p className="mt-3 truncate text-lg font-semibold text-foreground">{latestTraining?.experimentName || '--'}</p>
              <p className="mt-2 truncate font-mono text-xs text-muted-text">{latestTraining?.runId || '--'}</p>
            </Card>
            <Card variant="bordered" padding="md">
              <span className="label-uppercase">Status</span>
              <p className="mt-3 text-lg font-semibold text-foreground">{latestTraining?.status || '--'}</p>
              <p className="mt-2 text-xs text-muted-text">
                {formatUiText(text.qlibStatusCount, { count: String(status?.experimentNotebook?.totalRuns ?? 0) })}
              </p>
            </Card>
            <Card variant="bordered" padding="md">
              <span className="label-uppercase">Boundary</span>
              <p className="mt-3 text-sm font-semibold text-foreground">{text.qlibBoundaryTitle}</p>
              <p className="mt-2 text-xs leading-5 text-muted-text">{text.qlibBoundaryDescription}</p>
            </Card>
          </div>

          <Card variant="bordered" padding="md">
            <div className="flex items-center justify-between gap-3">
              <h3 className="text-sm font-semibold text-foreground">{text.qlibLatestMetrics}</h3>
              {latestTraining?.runId ? <Badge variant="info">{latestTraining.runId}</Badge> : null}
            </div>
            {metrics.length === 0 ? (
              <p className="mt-3 text-sm text-secondary-text">{text.qlibNoMetrics}</p>
            ) : (
              <div className="mt-3 grid gap-2 md:grid-cols-2">
                {metrics.map(([name, value]) => (
                  <div key={name} className="rounded-xl border border-subtle bg-surface/35 px-3 py-2">
                    <p className="truncate text-xs text-muted-text">{name}</p>
                    <p className="mt-1 font-mono text-sm text-foreground">
                      {typeof value === 'number' ? formatQlibMetric(value) : String(value)}
                    </p>
                  </div>
                ))}
              </div>
            )}
          </Card>

          <QlibPredictionCard report={qlibPredictions} text={text} />

          {candidatePipeline ? <CandidatePipelineCard report={candidatePipeline} text={text} /> : null}

          <RobustSignalCard report={status?.robustSignal} text={text} />

          <RobustStrategyCard report={status?.robustStrategy} text={text} />

          <QlibArtifactsCard status={status} text={text} />

          <CashControlBacktestCard report={cashControlBacktest} text={text} />
        </section>

        <aside className="flex flex-col gap-3">
          <Card variant="bordered" padding="md">
            <h3 className="text-sm font-semibold text-foreground">{text.qlibRecentRuns}</h3>
            <div className="mt-3 flex flex-col gap-3">
              {runs.length === 0 ? (
                <p className="text-sm text-secondary-text">{text.qlibNoRuns}</p>
              ) : runs.slice(0, 5).map((run, index) => (
                <QlibRunCard key={run.runId || `${run.experimentName || 'run'}-${index}`} run={run} text={text} />
              ))}
            </div>
          </Card>
        </aside>
      </div>
    </main>
  );
};

// ============ Main Page ============

const BacktestPage: React.FC = () => {
  const { language, t } = useUiLanguage();
  const text = BACKTEST_TEXT[language];
  const phaseFilterOptions = BACKTEST_PHASE_FILTER_OPTIONS[language];
  const actionLabels = buildDecisionActionLabelMap(t);
  const [activeTab, setActiveTab] = useState<BacktestTab>('dsa');
  const [researchStatus, setResearchStatus] = useState<ResearchStatus | null>(null);
  const [isLoadingQlibStatus, setIsLoadingQlibStatus] = useState(false);
  const [qlibStatusError, setQlibStatusError] = useState<string | null>(null);

  // Set page title
  useEffect(() => {
    document.title = text.documentTitle;
  }, [text.documentTitle]);

  useEffect(() => {
    if (activeTab !== 'qlib') {
      return undefined;
    }
    let active = true;
    setIsLoadingQlibStatus(true);
    researchApi.getStatus()
      .then((status) => {
        if (!active) return;
        setResearchStatus(status);
        setQlibStatusError(null);
      })
      .catch((err: unknown) => {
        if (!active) return;
        setQlibStatusError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (active) setIsLoadingQlibStatus(false);
      });
    return () => {
      active = false;
    };
  }, [activeTab]);

  // Input state
  const [codeFilter, setCodeFilter] = useState('');
  const [analysisDateFrom, setAnalysisDateFrom] = useState('');
  const [analysisDateTo, setAnalysisDateTo] = useState('');
  const [phaseFilter, setPhaseFilter] = useState<BacktestPhaseFilter>('all');
  const [evalDays, setEvalDays] = useState('');
  const [forceRerun, setForceRerun] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [runResult, setRunResult] = useState<BacktestRunResponse | null>(null);
  const [runError, setRunError] = useState<ParsedApiError | null>(null);
  const [pageError, setPageError] = useState<ParsedApiError | null>(null);

  // Results state
  const [results, setResults] = useState<BacktestResultItem[]>([]);
  const [totalResults, setTotalResults] = useState(0);
  const [currentPage, setCurrentPage] = useState(1);
  const [isLoadingResults, setIsLoadingResults] = useState(false);
  const pageSize = 20;

  // Performance state
  const [overallPerf, setOverallPerf] = useState<PerformanceMetrics | null>(null);
  const [stockPerf, setStockPerf] = useState<PerformanceMetrics | null>(null);
  const [isLoadingPerf, setIsLoadingPerf] = useState(false);
  const effectiveWindowDays = evalDays ? parseInt(evalDays, 10) : overallPerf?.evalWindowDays;
  const isNextDayValidation = effectiveWindowDays === 1;
  const showNextDayActualColumns = isNextDayValidation;

  // Fetch results
  const fetchResults = useCallback(async (
    page = 1,
    code?: string,
    windowDays?: number,
    startDate?: string,
    endDate?: string,
    phase?: BacktestPhaseFilter,
  ) => {
    setIsLoadingResults(true);
    try {
      const response = await backtestApi.getResults({
        code: code || undefined,
        evalWindowDays: windowDays,
        analysisDateFrom: startDate || undefined,
        analysisDateTo: endDate || undefined,
        analysisPhase: phase && phase !== 'all' ? phase : undefined,
        page,
        limit: pageSize,
      });
      setResults(response.items);
      setTotalResults(response.total);
      setCurrentPage(response.page);
      setPageError(null);
    } catch (err) {
      console.error('Failed to fetch backtest results:', err);
      setPageError(getParsedApiError(err));
    } finally {
      setIsLoadingResults(false);
    }
  }, []);

  // Fetch performance
  const fetchPerformance = useCallback(async (
    code?: string,
    windowDays?: number,
    startDate?: string,
    endDate?: string,
    phase?: BacktestPhaseFilter,
  ) => {
    setIsLoadingPerf(true);
    try {
      const overall = await backtestApi.getOverallPerformance({
        evalWindowDays: windowDays,
        analysisDateFrom: startDate || undefined,
        analysisDateTo: endDate || undefined,
        analysisPhase: phase && phase !== 'all' ? phase : undefined,
      });
      setOverallPerf(overall);

      if (code) {
        const stock = await backtestApi.getStockPerformance(code, {
          evalWindowDays: windowDays,
          analysisDateFrom: startDate || undefined,
          analysisDateTo: endDate || undefined,
          analysisPhase: phase && phase !== 'all' ? phase : undefined,
        });
        setStockPerf(stock);
      } else {
        setStockPerf(null);
      }
      setPageError(null);
    } catch (err) {
      console.error('Failed to fetch performance:', err);
      setPageError(getParsedApiError(err));
    } finally {
      setIsLoadingPerf(false);
    }
  }, []);

  // Initial load — fetch performance first, then filter results by its window
  useEffect(() => {
    let active = true;
    const init = async () => {
      try {
        // Get latest performance (unfiltered returns most recent summary)
        const overall = await backtestApi.getOverallPerformance();
        if (!active) return;
        setOverallPerf(overall);
        // Use the summary's eval_window_days to filter results consistently
        const windowDays = overall?.evalWindowDays;
        if (windowDays && !evalDays) {
          setEvalDays(String(windowDays));
        }
        fetchResults(1, undefined, windowDays, undefined, undefined, 'all');
      } catch (err) {
        if (!active) return;
        console.error('Failed to initialize backtest page:', err);
        setPageError(getParsedApiError(err));
        setResults([]);
        setTotalResults(0);
      }
    };
    void init();
    return () => {
      active = false;
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Run backtest
  const handleRun = async () => {
    setIsRunning(true);
    setRunResult(null);
    setRunError(null);
    try {
      const code = codeFilter.trim() || undefined;
      const evalWindowDays = evalDays ? parseInt(evalDays, 10) : undefined;
      const response = await backtestApi.run({
        code,
        force: forceRerun || undefined,
        minAgeDays: forceRerun ? 0 : undefined,
        evalWindowDays,
      });
      setRunResult(response);
      // Refresh data with same eval_window_days
      fetchResults(1, codeFilter.trim() || undefined, evalWindowDays, analysisDateFrom, analysisDateTo, phaseFilter);
      fetchPerformance(codeFilter.trim() || undefined, evalWindowDays, analysisDateFrom, analysisDateTo, phaseFilter);
    } catch (err) {
      setRunError(getParsedApiError(err));
    } finally {
      setIsRunning(false);
    }
  };

  // Filter by code
  const handleFilter = () => {
    const code = codeFilter.trim() || undefined;
    const windowDays = evalDays ? parseInt(evalDays, 10) : undefined;
    setCurrentPage(1);
    fetchResults(1, code, windowDays, analysisDateFrom, analysisDateTo, phaseFilter);
    fetchPerformance(code, windowDays, analysisDateFrom, analysisDateTo, phaseFilter);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleFilter();
    }
  };

  const handleShowNextDay = () => {
    const code = codeFilter.trim() || undefined;
    setEvalDays('1');
    setCurrentPage(1);
    fetchResults(1, code, 1, analysisDateFrom, analysisDateTo, phaseFilter);
    fetchPerformance(code, 1, analysisDateFrom, analysisDateTo, phaseFilter);
  };

  // Pagination
  const totalPages = Math.ceil(totalResults / pageSize);
  const handlePageChange = (page: number) => {
    const windowDays = evalDays ? parseInt(evalDays, 10) : undefined;
    fetchResults(page, codeFilter.trim() || undefined, windowDays, analysisDateFrom, analysisDateTo, phaseFilter);
  };

  return (
    <div className="min-h-full flex flex-col rounded-[1.5rem] bg-transparent">
      <div className="flex flex-wrap gap-2 border-b border-white/5 px-3 py-3 sm:px-4">
        <button
          type="button"
          className={`btn-secondary h-9 px-3 text-sm ${activeTab === 'dsa' ? 'border-cyan/40 bg-cyan/10 text-cyan' : ''}`}
          onClick={() => setActiveTab('dsa')}
        >
          {text.tabDsa}
        </button>
        <button
          type="button"
          className={`btn-secondary h-9 px-3 text-sm ${activeTab === 'qlib' ? 'border-cyan/40 bg-cyan/10 text-cyan' : ''}`}
          onClick={() => setActiveTab('qlib')}
        >
          {text.tabQlib}
        </button>
        <Link
          to="/backtest/etf-momentum"
          className="btn-secondary h-9 px-3 text-sm"
        >
          {language === 'zh' ? 'ETF 动量完整报告' : 'ETF momentum report'}
        </Link>
      </div>

      {activeTab === 'qlib' ? (
        <QlibBacktestPanel
          status={researchStatus}
          loading={isLoadingQlibStatus}
          error={qlibStatusError}
          text={text}
        />
      ) : (
      <>
      {/* Header */}
      <header className="flex-shrink-0 border-b border-white/5 px-3 py-3 sm:px-4">
        <div className="flex max-w-5xl flex-wrap items-center gap-2">
          <div className="relative min-w-0 flex-[1_1_220px]">
            <input
              type="text"
              value={codeFilter}
              onChange={(e) => setCodeFilter(e.target.value.toUpperCase())}
              onKeyDown={handleKeyDown}
              placeholder={text.codePlaceholder}
              disabled={isRunning}
              className={BACKTEST_INPUT_CLASS}
            />
          </div>
          <button
            type="button"
            onClick={handleFilter}
            disabled={isLoadingResults}
            className="btn-secondary flex items-center gap-1.5 whitespace-nowrap"
          >
            {text.filter}
          </button>
          <div className="flex items-center gap-2 whitespace-nowrap lg:w-40 lg:justify-between">
            <span className="text-xs text-muted-text">{text.evalWindow}</span>
            <input
              type="number"
              min={1}
              max={120}
              value={evalDays}
              onChange={(e) => setEvalDays(e.target.value)}
              placeholder="10"
              disabled={isRunning}
              className={`${BACKTEST_COMPACT_INPUT_CLASS} w-24 text-center tabular-nums`}
            />
          </div>
          <div className="flex items-center gap-2 whitespace-nowrap">
            <span className="text-xs text-muted-text">{text.phase}</span>
            <select
              value={phaseFilter}
              onChange={(e) => setPhaseFilter(e.target.value as BacktestPhaseFilter)}
              disabled={isRunning}
              className={`${BACKTEST_COMPACT_INPUT_CLASS} w-28`}
            >
              {phaseFilterOptions.map((option) => (
                <option key={option.value} value={option.value}>{option.label}</option>
              ))}
            </select>
          </div>
          <div className="flex items-center gap-2 whitespace-nowrap">
            <span className="text-xs text-muted-text">{text.startDate}</span>
            <input
              type="date"
              aria-label={text.startDateAria}
              value={analysisDateFrom}
              onChange={(e) => setAnalysisDateFrom(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isRunning}
              className={`${BACKTEST_COMPACT_INPUT_CLASS} w-40 text-center tabular-nums`}
            />
          </div>
          <div className="flex items-center gap-2 whitespace-nowrap">
            <span className="text-xs text-muted-text">{text.endDate}</span>
            <input
              type="date"
              aria-label={text.endDateAria}
              value={analysisDateTo}
              onChange={(e) => setAnalysisDateTo(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isRunning}
              className={`${BACKTEST_COMPACT_INPUT_CLASS} w-40 text-center tabular-nums`}
            />
          </div>
          <button
            type="button"
            onClick={handleShowNextDay}
            disabled={isLoadingResults || isLoadingPerf}
            className={`backtest-force-btn ${isNextDayValidation ? 'active' : ''}`}
          >
            <span className="dot" />
            {text.oneDayValidation}
          </button>
          <button
            type="button"
            onClick={() => setForceRerun(!forceRerun)}
            disabled={isRunning}
            className={`backtest-force-btn ${forceRerun ? 'active' : ''}`}
          >
            <span className="dot" />
            {text.forceRerun}
          </button>
          <button
            type="button"
            onClick={handleRun}
            disabled={isRunning}
            className="btn-primary flex items-center gap-1.5 whitespace-nowrap"
          >
            {isRunning ? (
              <>
                <svg className="w-3.5 h-3.5 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                {text.running}
              </>
            ) : (
              text.runBacktest
            )}
          </button>
        </div>
        {runResult && (
          <div className="mt-2 max-w-4xl">
            <RunSummary data={runResult} language={language} />
          </div>
        )}
        {runError && (
          <ApiErrorAlert error={runError} className="mt-2 max-w-4xl" />
        )}
        <p className="mt-2 text-xs text-muted-text">
          {isNextDayValidation
            ? text.oneDayModeDescription
            : text.windowModeDescription}
        </p>
      </header>

      {/* Main content */}
      <main className="flex min-h-0 flex-1 flex-col gap-3 overflow-hidden p-3 lg:flex-row">
        {/* Left sidebar - Performance */}
        <div className="flex max-h-[38vh] flex-col gap-3 overflow-y-auto lg:max-h-none lg:w-60 lg:flex-shrink-0">
          {isLoadingPerf ? (
            <div className="flex items-center justify-center py-8">
              <div className="backtest-spinner sm" />
            </div>
          ) : overallPerf ? (
            <PerformanceCard metrics={overallPerf} title={text.overallPerformance} language={language} />
          ) : (
            <EmptyState
              title={text.noMetricsTitle}
              description={text.noMetricsDescription}
              className="h-full min-h-[12rem] border-dashed bg-card/45 shadow-none"
            />
          )}

          {stockPerf && (
            <PerformanceCard metrics={stockPerf} title={`${stockPerf.code || codeFilter}`} language={language} />
          )}
        </div>

        {/* Right content - Results table */}
        <section className="min-h-0 flex-1 overflow-y-auto">
          {pageError ? (
            <ApiErrorAlert error={pageError} className="mb-3" />
          ) : null}
          {isLoadingResults ? (
            <div className="flex flex-col items-center justify-center h-64">
              <div className="backtest-spinner md" />
              <p className="mt-3 text-secondary-text text-sm">{text.loadingResults}</p>
            </div>
          ) : results.length === 0 ? (
            <EmptyState
              title={text.noResultsTitle}
              description={text.noResultsDescription}
              className="backtest-empty-state border-dashed"
              icon={(
                <svg className="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                </svg>
              )}
            />
          ) : (
            <div className="animate-fade-in">
              <div className="backtest-table-toolbar">
                <div className="backtest-table-toolbar-meta">
                  <span className="label-uppercase">{isNextDayValidation ? text.nextDayValidation : text.resultSet}</span>
                  <span className="text-xs text-secondary-text">
                    {codeFilter.trim() ? formatUiText(text.filteredStock, { code: codeFilter.trim() }) : text.allStocks}
                    {evalDays ? ` · ${formatUiText(text.dayWindow, { days: evalDays })}` : ''}
                    {phaseFilter !== 'all' ? ` · ${phaseFilterOptions.find((item) => item.value === phaseFilter)?.label ?? phaseFilter}` : ''}
                    {analysisDateFrom ? ` · ${formatUiText(text.fromDate, { date: analysisDateFrom })}` : ''}
                    {analysisDateTo ? ` · ${formatUiText(text.toDate, { date: analysisDateTo })}` : ''}
                  </span>
                </div>
                <span className="backtest-table-scroll-hint">{text.scrollHint}</span>
              </div>
              <div className="backtest-table-wrapper">
                <table className="backtest-table min-w-[900px] w-full text-sm">
                  <thead className="backtest-table-head">
                    <tr className="text-left">
                      <th className="backtest-table-head-cell">{text.stock}</th>
                      <th className="backtest-table-head-cell">{text.analysisDate}</th>
                      <th className="backtest-table-head-cell">{text.phase}</th>
                      <th className="backtest-table-head-cell">{text.aiPrediction}</th>
                      <th className="backtest-table-head-cell">
                        {showNextDayActualColumns ? text.actualPerformance : text.windowReturn}
                      </th>
                      <th className="backtest-table-head-cell">
                        {showNextDayActualColumns ? text.accuracy : text.directionMatch}
                      </th>
                      <th className="backtest-table-head-cell">{text.result}</th>
                      <th className="backtest-table-head-cell">{text.status}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.map((row) => {
                      const actionLabel = getDecisionActionLabel(row.action, row.actionLabel, null, null, actionLabels);
                      const predictionParts = [actionLabel, row.trendPrediction, row.operationAdvice]
                        .filter((part): part is string => Boolean(part));

                      return (
                        <tr
                          key={row.analysisHistoryId}
                          className="backtest-table-row"
                        >
                          <td className="backtest-table-cell backtest-table-code">
                            <div className="flex flex-col">
                              <span>{row.code}</span>
                              <span className="text-xs text-muted-text">{row.stockName || '--'}</span>
                            </div>
                          </td>
                          <td className="backtest-table-cell text-secondary-text">{row.analysisDate || '--'}</td>
                          <td className="backtest-table-cell text-secondary-text">{phaseLabel(row, language)}</td>
                          <td className="backtest-table-cell max-w-[220px] text-foreground">
                            {predictionParts.length ? (
                              <Tooltip
                                content={predictionParts.join(' / ')}
                                focusable
                              >
                                <div className="flex flex-col gap-1">
                                  <span className="block truncate">{actionLabel || row.trendPrediction || '--'}</span>
                                  {actionLabel && row.trendPrediction && (
                                    <span className="block truncate text-xs text-secondary-text">{row.trendPrediction}</span>
                                  )}
                                  {row.operationAdvice && (
                                    <span className="block truncate text-xs text-secondary-text">{row.operationAdvice}</span>
                                  )}
                                </div>
                              </Tooltip>
                            ) : (
                              '--'
                            )}
                          </td>
                          <td className="backtest-table-cell">
                            <div className="flex items-center gap-2">
                              {actualMovementBadge(row.actualMovement, language)}
                              <span className={
                                row.actualReturnPct != null
                                  ? row.actualReturnPct > 0 ? 'text-success' : row.actualReturnPct < 0 ? 'text-danger' : 'text-secondary-text'
                                  : 'text-muted-text'
                              }>
                                {pct(row.actualReturnPct)}
                              </span>
                            </div>
                          </td>
                          <td className="backtest-table-cell">
                            <span className="flex items-center gap-2">
                              {boolIcon(row.directionCorrect, text)}
                              <span className="text-muted-text">
                                {row.directionExpected ? labelFromMap(row.directionExpected, BACKTEST_DIRECTION_EXPECTED_LABELS[language]) : ''}
                              </span>
                            </span>
                          </td>
                          <td className="backtest-table-cell">{outcomeBadge(row.outcome, language)}</td>
                          <td className="backtest-table-cell">{statusBadge(row.evalStatus, language)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              {/* Pagination */}
              <div className="mt-4">
                <Pagination
                  currentPage={currentPage}
                  totalPages={totalPages}
                  onPageChange={handlePageChange}
                />
              </div>

              <p className="text-xs text-muted-text text-center mt-2">
                {formatUiText(text.totalPage, { total: totalResults, page: currentPage, pages: Math.max(totalPages, 1) })}
              </p>
            </div>
          )}
        </section>
      </main>
      </>
      )}
    </div>
  );
};

export default BacktestPage;
