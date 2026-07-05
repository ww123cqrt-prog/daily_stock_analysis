import type React from 'react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Activity,
  ArrowRight,
  ArrowUpRight,
  CheckCircle2,
  CircleStop,
  Database,
  Eye,
  FileText,
  FlaskConical,
  History,
  ListChecks,
  Loader2,
  Play,
  RefreshCw,
  RotateCw,
  TerminalSquare,
} from 'lucide-react';
import { Badge, Card, InlineAlert, StatusDot } from '../components/common';
import { useUiLanguage } from '../contexts/UiLanguageContext';
import { getParsedApiError } from '../api/error';
import { parseQuantJobStreamEvent, quantApi, type QuantJob, type QuantStatus } from '../api/quant';
import { researchApi, type ResearchStatus } from '../api/research';

type WorkflowJob = {
  type: string;
  label: string;
  description: string;
  defaultParams?: Record<string, unknown>;
  params?: WorkflowParamField[];
};

type WorkflowParamField = {
  key: string;
  label: string;
  kind: 'text' | 'number' | 'date' | 'checkbox' | 'select';
  placeholder?: string;
  min?: number;
  max?: number;
  step?: number;
  options?: Array<{ value: string; label: string }>;
};

type WorkflowStep = {
  title: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  jobs: WorkflowJob[];
};

type DailyGuideStep = {
  id: string;
  title: string;
  detail: string;
  status: 'done' | 'running' | 'failed' | 'todo' | 'waiting';
  statusLabel: string;
  statusTone: 'success' | 'danger' | 'warning' | 'info' | 'neutral';
  meta?: string;
  actions: DailyGuideAction[];
};

type DailyGuideAction =
  | {
      kind: 'job';
      label: string;
      jobType: string;
      params?: Record<string, unknown>;
    }
  | {
      kind: 'link';
      label: string;
      to: string;
    }
  | {
      kind: 'external';
      label: string;
      href: string;
    };

const STREAM_RECONNECT_INITIAL_DELAY_MS = 1000;
const STREAM_RECONNECT_MAX_DELAY_MS = 15000;
const WORKFLOW_INPUT_CLASS =
  'input-surface input-focus-glow h-9 w-full rounded-lg border bg-transparent px-2.5 text-xs transition-all focus:outline-none';

const WORKFLOW_STEPS: WorkflowStep[] = [
  {
    title: '更新数据',
    description: '准备增量清单，并从 Baostock/JQData 拉取本地价格数据。',
    icon: Database,
    jobs: [
      {
        type: 'prepare_baostock_refresh',
        label: '准备 Baostock 增量清单',
        description: '根据异常清单和目标日期生成需要补拉的股票列表。',
        params: [
          { key: 'target_end', label: '目标结束日', kind: 'date' },
          { key: 'limit', label: '股票数量上限', kind: 'number', min: 1, placeholder: '留空为全部' },
        ],
      },
      {
        type: 'download_baostock',
        label: '下载 Baostock 长历史',
        description: '补齐 A 股日线长历史，本地缓存为 Parquet。',
        params: [
          { key: 'start', label: '开始日期', kind: 'date' },
          { key: 'end', label: '结束日期', kind: 'date' },
          { key: 'limit', label: '股票数量上限', kind: 'number', min: 1 },
          { key: 'sleep', label: '请求间隔秒', kind: 'number', min: 0, step: 0.1, placeholder: '0.2' },
          { key: 'retries', label: '重试次数', kind: 'number', min: 0 },
          { key: 'timeout', label: '超时秒数', kind: 'number', min: 1 },
          { key: 'max_consecutive_failures', label: '连续失败上限', kind: 'number', min: 1 },
          { key: 'incremental', label: '增量下载', kind: 'checkbox' },
          { key: 'retry_failed', label: '重试失败', kind: 'checkbox' },
          { key: 'retry_empty', label: '重试空数据', kind: 'checkbox' },
          { key: 'retry_timeout', label: '重试超时', kind: 'checkbox' },
        ],
      },
      {
        type: 'download_jqdata',
        label: '验证 JQData 连接',
        description: '只做认证、股票池解析和小样本 dry-run，不写入真实 Parquet 数据目录。',
        defaultParams: { dry_run: true },
        params: [
          { key: 'universe', label: '股票池', kind: 'text', placeholder: 'all / csi300 / 000001,600519' },
          { key: 'start', label: '开始日期', kind: 'date' },
          { key: 'end', label: '结束日期', kind: 'date' },
          { key: 'limit', label: '验证样本数', kind: 'number', min: 1, max: 20, placeholder: '默认3，最多20' },
          { key: 'sleep', label: '请求间隔秒', kind: 'number', min: 0, step: 0.1 },
        ],
      },
    ],
  },
  {
    title: '合并 Qlib',
    description: '把多来源 Parquet 合并，并转换成 Qlib provider。',
    icon: RotateCw,
    jobs: [
      {
        type: 'merge_to_qlib',
        label: '合并并转换为 Qlib',
        description: '生成 canonical Qlib provider，供模型训练读取。',
        params: [
          { key: 'smoke', label: '使用隔离小样本演练', kind: 'checkbox' },
          { key: 'no_overwrite', label: '跳过已有输出文件', kind: 'checkbox' },
        ],
      },
      {
        type: 'repair_qlib_provider',
        label: '修复 Qlib Provider',
        description: '修复日历、股票池、benchmark 等 provider 元数据。',
        params: [
          { key: 'dry_run', label: '只检查不写入', kind: 'checkbox' },
        ],
      },
      { type: 'check_qlib_data', label: '检查 Qlib 数据', description: '快速读取样本股票和字段，确认 provider 可用。' },
    ],
  },
  {
    title: '质量报告',
    description: '生成数据质量、异常股票和清洗策略报告。',
    icon: FileText,
    jobs: [
      {
        type: 'report_data_quality',
        label: '生成数据质量报告',
        description: '汇总覆盖率、日期范围、来源重叠和可训练状态。',
        params: [
          { key: 'overlap_sample', label: '重叠样本数', kind: 'number', min: 1 },
          { key: 'merged_sample', label: '合并样本数', kind: 'number', min: 1 },
        ],
      },
      { type: 'report_data_anomalies', label: '生成异常股票和清洗策略', description: '输出异常股票清单，并给出后续清洗策略。' },
      {
        type: 'run_china_astock_comparison',
        label: '生成 china-astock 策略对比',
        description: '读取 china-astock-quant 既有搜索和赢家评估结果，生成策略对比 CSV/Markdown 报告；只读报告，不连接券商或交易接口。',
      },
      {
        type: 'apply_data_cleaning',
        label: '应用清洗策略',
        description: '根据异常清单生成 cleaned Qlib provider，不覆盖原始 canonical provider。',
        params: [
          { key: 'dry_run', label: '只预演不写入', kind: 'checkbox' },
          { key: 'no_overwrite', label: '不覆盖已有文件', kind: 'checkbox' },
          { key: 'skip_qlib_convert', label: '仅生成 cleaned Parquet', kind: 'checkbox' },
        ],
      },
    ],
  },
  {
    title: '训练+回测',
    description: '运行 Qlib Alpha158 + LightGBM baseline，并记录 MLflow 实验。',
    icon: FlaskConical,
    jobs: [
      {
        type: 'train_baseline_lgb',
        label: '训练 Qlib LightGBM 基线',
        description: '执行 Qlib 训练和组合回测，写入最新训练摘要。',
        params: [
          {
            key: 'provider',
            label: '数据版本',
            kind: 'select',
            options: [
              { value: 'canonical', label: '原始合并数据' },
              { value: 'cleaned', label: '清洗后数据' },
            ],
          },
          { key: 'config', label: '配置文件', kind: 'text', placeholder: 'baseline_lgb_alpha158.yaml' },
          { key: 'experiment_name', label: '实验名称', kind: 'text', placeholder: 'baseline_lgb_alpha158' },
        ],
      },
      {
        type: 'report_market_regime',
        label: '生成市场状态和现金仓位信号',
        description: '计算等权市场收益、市场宽度、回撤和波动率，输出 risk_on/neutral/defensive/risk_off 与建议现金比例；只读研究报告，不连接券商。',
        params: [
          { key: 'lookback_days', label: '回看交易日', kind: 'number', min: 2, placeholder: '252' },
          { key: 'min_symbols', label: '最少股票数', kind: 'number', min: 1, placeholder: '100' },
          { key: 'limit', label: '抽样上限', kind: 'number', min: 1, placeholder: '留空为全量' },
        ],
      },
      {
        type: 'report_cash_control_backtest',
        label: '生成现金仓位叠加回测',
        description: '读取最新 Qlib 组合回测日度收益，用市场状态信号叠加现金仓位，输出 baseline vs cash-control 对比；只读报告，不连接券商。',
        params: [
          { key: 'lookback_days', label: '回看交易日', kind: 'number', min: 2, placeholder: '60' },
          { key: 'min_symbols', label: '最少股票数', kind: 'number', min: 1, placeholder: '100' },
          { key: 'limit', label: '抽样上限', kind: 'number', min: 1, placeholder: '留空为全量' },
        ],
      },
      {
        type: 'report_qlib_predictions',
        label: '生成 Qlib 预测摘要',
        description: '读取最新 Qlib/MLflow pred.pkl，生成最新交易日 Top/Bottom 模型评分摘要；只读排行，不连接券商、不自动下单。',
        defaultParams: { topk: '20' },
        params: [
          { key: 'topk', label: 'TopK', kind: 'number', min: 1, max: 500, placeholder: '20' },
          { key: 'run_id', label: 'MLflow Run ID', kind: 'text', placeholder: '留空读取最新训练' },
        ],
      },
      {
        type: 'optimize_strategy',
        label: '策略优化预演',
        description: '用内置小样本 dry-run 验证低换手、市场择时和回撤保护实验链路；只读生成报告，不训练、不连接券商、不自动下单。',
        defaultParams: { dry_run: true, phase: '4', topk: '30' },
        params: [
          {
            key: 'phase',
            label: '阶段',
            kind: 'select',
            options: [
              { value: '0', label: '全部阶段' },
              { value: '1', label: '低换手' },
              { value: '2', label: '市场择时' },
              { value: '3', label: '回撤保护' },
              { value: '4', label: '综合策略' },
            ],
          },
          { key: 'topk', label: 'TopK', kind: 'number', min: 1, max: 500, placeholder: '30' },
        ],
      },
      {
        type: 'run_dsa_backtest',
        label: '运行 DSA 分析准确性回测',
        description: '验证 DSA 历史分析建议和后续走势的一致性。',
        params: [
          { key: 'code', label: '股票代码', kind: 'text', placeholder: '留空为全部' },
          { key: 'eval_window_days', label: '评估窗口', kind: 'number', min: 1, placeholder: '1' },
          { key: 'force', label: '强制重跑', kind: 'checkbox' },
        ],
      },
    ],
  },
];

const WORKFLOW_JOBS_BY_TYPE = Object.fromEntries(
  WORKFLOW_STEPS.flatMap((step) => step.jobs).map((job) => [job.type, job]),
);

type JobParamValue = string | boolean;
type JobParamState = Record<string, Record<string, JobParamValue>>;

const STATUS_LABEL: Record<string, string> = {
  pending: '等待中',
  running: '运行中',
  completed: '完成',
  failed: '失败',
  cancelled: '已取消',
};

function statusTone(status: string): 'success' | 'danger' | 'warning' | 'info' | 'neutral' {
  if (status === 'completed') return 'success';
  if (status === 'failed') return 'danger';
  if (status === 'cancelled') return 'warning';
  if (status === 'running' || status === 'pending') return 'info';
  return 'neutral';
}

function formatTime(value?: string | null): string {
  if (!value) return '-';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function formatNumber(value?: number): string {
  if (value === undefined || Number.isNaN(value)) return '-';
  return new Intl.NumberFormat('zh-CN').format(value);
}

function artifactFileName(path: string): string {
  return path.split(/[\\/]/).filter(Boolean).pop() || path;
}

function jobTimestamp(job: QuantJob): number {
  const timestamp = new Date(job.updatedAt || job.createdAt || '').getTime();
  return Number.isNaN(timestamp) ? 0 : timestamp;
}

function latestJobsByType(jobs: QuantJob[]): Record<string, QuantJob> {
  const latest: Record<string, QuantJob> = {};
  for (const job of jobs) {
    const current = latest[job.jobType];
    if (!current || jobTimestamp(job) >= jobTimestamp(current)) {
      latest[job.jobType] = job;
    }
  }
  return latest;
}

function activeJobCount(jobs: QuantJob[]): number {
  return jobs.filter((job) => job.status === 'pending' || job.status === 'running').length;
}

function isActiveJob(job?: QuantJob): boolean {
  return job?.status === 'pending' || job?.status === 'running';
}

function isDoneJob(job?: QuantJob): boolean {
  return job?.status === 'completed';
}

function latestJobForTypes(latestJobs: Record<string, QuantJob>, jobTypes: string[]): QuantJob | undefined {
  return jobTypes
    .map((jobType) => latestJobs[jobType])
    .filter(Boolean)
    .sort((left, right) => jobTimestamp(right) - jobTimestamp(left))[0];
}

function jobDrivenStatus(
  job: QuantJob | undefined,
  fallbackStatus: DailyGuideStep['status'],
  fallbackLabel: string,
): Pick<DailyGuideStep, 'status' | 'statusLabel' | 'statusTone' | 'meta'> {
  if (!job) {
    return { status: fallbackStatus, statusLabel: fallbackLabel, statusTone: fallbackStatus === 'waiting' ? 'neutral' : 'info' };
  }
  if (job.status === 'running' || job.status === 'pending') {
    return { status: 'running', statusLabel: '运行中', statusTone: 'info', meta: formatTime(job.updatedAt) };
  }
  if (job.status === 'failed') {
    return { status: 'failed', statusLabel: '失败', statusTone: 'danger', meta: job.error || formatTime(job.updatedAt) };
  }
  if (job.status === 'cancelled') {
    return { status: 'waiting', statusLabel: '已取消', statusTone: 'warning', meta: formatTime(job.updatedAt) };
  }
  if (job.status === 'completed') {
    return { status: 'done', statusLabel: '已完成', statusTone: 'success', meta: formatTime(job.updatedAt) };
  }
  return { status: fallbackStatus, statusLabel: fallbackLabel, statusTone: 'neutral', meta: formatTime(job.updatedAt) };
}

function buildDailyGuideSteps(
  researchStatus: ResearchStatus | null,
  latestJobs: Record<string, QuantJob>,
): DailyGuideStep[] {
  const updateJob = latestJobForTypes(latestJobs, ['prepare_baostock_refresh', 'download_baostock', 'download_jqdata']);
  const qualityJob = latestJobs.report_data_quality;
  const anomalyJob = latestJobs.report_data_anomalies;
  const cleaningJob = latestJobs.apply_data_cleaning;
  const reviewJob = latestJobForTypes(latestJobs, ['report_market_regime', 'report_cash_control_backtest', 'train_baseline_lgb', 'run_dsa_backtest']);

  const dataStatus = researchStatus?.trainable
    ? {
        status: 'done' as const,
        statusLabel: '可训练',
        statusTone: 'success' as const,
        meta: researchStatus.qlibProvider?.calendarRange ? `Qlib: ${researchStatus.qlibProvider.calendarRange}` : undefined,
      }
    : jobDrivenStatus(latestJobs.check_qlib_data, 'todo', '待检查');

  const updateStatus = jobDrivenStatus(updateJob, 'waiting', '按需更新');

  const qualityStatus = (() => {
    const running = [qualityJob, anomalyJob].find(isActiveJob);
    if (running) return jobDrivenStatus(running, 'todo', '待检查');
    const failed = [qualityJob, anomalyJob].find((job) => job?.status === 'failed');
    if (failed) return jobDrivenStatus(failed, 'todo', '待检查');
    if (isDoneJob(qualityJob) && isDoneJob(anomalyJob)) {
      return { status: 'done' as const, statusLabel: '报告齐全', statusTone: 'success' as const, meta: formatTime(anomalyJob?.updatedAt || qualityJob?.updatedAt) };
    }
    if (isDoneJob(qualityJob) || isDoneJob(anomalyJob)) {
      return { status: 'waiting' as const, statusLabel: '部分完成', statusTone: 'warning' as const, meta: formatTime((qualityJob || anomalyJob)?.updatedAt) };
    }
    return { status: 'todo' as const, statusLabel: '待生成', statusTone: 'info' as const };
  })();

  const cleaningStatus = (() => {
    const base = jobDrivenStatus(cleaningJob, 'waiting', '建议先预演');
    if (cleaningJob?.status === 'completed' && cleaningJob.params?.dry_run === true) {
      return { ...base, statusLabel: '预演完成' };
    }
    if (cleaningJob?.status === 'completed') {
      return { ...base, statusLabel: '已应用' };
    }
    return base;
  })();

  const reviewStatus = (() => {
    const active = [latestJobs.report_market_regime, latestJobs.report_cash_control_backtest, latestJobs.train_baseline_lgb, latestJobs.run_dsa_backtest].find(isActiveJob);
    if (active) return jobDrivenStatus(active, 'todo', '待运行');
    const failed = [latestJobs.report_market_regime, latestJobs.report_cash_control_backtest, latestJobs.train_baseline_lgb, latestJobs.run_dsa_backtest].find((job) => job?.status === 'failed');
    if (failed) return jobDrivenStatus(failed, 'todo', '待运行');
    if (researchStatus?.latestTraining?.status === 'completed' || isDoneJob(reviewJob)) {
      return { status: 'done' as const, statusLabel: '有结果', statusTone: 'success' as const, meta: researchStatus?.latestTraining?.experimentName || formatTime(reviewJob?.updatedAt) };
    }
    return { status: 'todo' as const, statusLabel: '待运行', statusTone: 'info' as const };
  })();

  const noteStatus = (researchStatus?.experimentNotebook?.totalRuns ?? 0) > 0
    ? {
        status: 'done' as const,
        statusLabel: `${researchStatus?.experimentNotebook?.totalRuns ?? 0} 条记录`,
        statusTone: 'success' as const,
        meta: researchStatus?.experimentNotebook?.latestNote,
      }
    : {
        status: 'todo' as const,
        statusLabel: '待记录',
        statusTone: 'info' as const,
      };

  return [
    {
      id: 'check-data',
      title: '检查数据',
      detail: '确认 Qlib provider 可读、日期范围有效、当前数据是否可训练。',
      ...dataStatus,
      actions: [
        { kind: 'job', label: '检查 Qlib', jobType: 'check_qlib_data' },
        { kind: 'link', label: '研究总览', to: '/research' },
      ],
    },
    {
      id: 'refresh-data',
      title: '更新数据',
      detail: '日常先生成增量清单；真正下载长任务仍在左侧填写参数后运行。',
      ...updateStatus,
      actions: [
        { kind: 'job', label: '增量清单', jobType: 'prepare_baostock_refresh' },
      ],
    },
    {
      id: 'quality',
      title: '质量和异常',
      detail: '生成覆盖率、来源重叠、异常股票清单和清洗策略报告。',
      ...qualityStatus,
      actions: [
        { kind: 'job', label: '质量报告', jobType: 'report_data_quality' },
        { kind: 'job', label: '异常清单', jobType: 'report_data_anomalies' },
      ],
    },
    {
      id: 'cleaning',
      title: '清洗预演',
      detail: '默认只做 dry-run，不写 cleaned parquet，不转换 Qlib，适合先核对影响范围。',
      ...cleaningStatus,
      actions: [
        { kind: 'job', label: '预演', jobType: 'apply_data_cleaning', params: { dry_run: true, no_overwrite: true } },
      ],
    },
    {
      id: 'review',
      title: '训练/复盘',
      detail: '跑 Qlib LightGBM 基线、现金仓位叠加或 DSA 历史准确性回测，然后进入回测页看结果。',
      ...reviewStatus,
      actions: [
        { kind: 'job', label: '市场信号', jobType: 'report_market_regime' },
        { kind: 'job', label: '现金叠加', jobType: 'report_cash_control_backtest' },
        { kind: 'job', label: '训练基线', jobType: 'train_baseline_lgb' },
        { kind: 'link', label: '回测页', to: '/backtest' },
      ],
    },
    {
      id: 'notes',
      title: '记录结论',
      detail: '把数据版本、指标、结论和下一步留在实验记录本，便于回看和删除。',
      ...noteStatus,
      actions: [
        { kind: 'link', label: '实验记录', to: '/research' },
        { kind: 'external', label: 'MLflow', href: 'http://127.0.0.1:6006/' },
      ],
    },
  ];
}

function workflowErrorMessage(err: unknown): string {
  const parsed = getParsedApiError(err);
  return parsed.message || parsed.rawMessage || '请求未成功完成，请稍后重试。';
}

function buildJobParams(job: WorkflowJob, values: Record<string, JobParamValue> = {}): Record<string, unknown> {
  const params: Record<string, unknown> = { ...(job.defaultParams || {}) };
  for (const field of job.params || []) {
    const value = values[field.key];
    if (field.kind === 'checkbox') {
      if (value === true) params[field.key] = true;
      continue;
    }
    if (typeof value === 'string' && value.trim()) {
      params[field.key] = value.trim();
    }
  }
  return params;
}

const JobStatusBadge: React.FC<{ job?: QuantJob }> = ({ job }) => {
  if (!job) {
    return (
      <Badge variant="default" size="sm" className="gap-1.5 shadow-none">
        <StatusDot tone="neutral" className="h-1.5 w-1.5" />
        未运行
      </Badge>
    );
  }
  const tone = statusTone(job.status);
  return (
    <Badge variant={tone === 'success' ? 'success' : tone === 'danger' ? 'danger' : tone === 'warning' ? 'warning' : 'info'} size="sm" className="gap-1.5 shadow-none">
      <StatusDot tone={tone} pulse={job.status === 'running'} className="h-1.5 w-1.5" />
      {STATUS_LABEL[job.status] || job.status}
    </Badge>
  );
};

const WorkflowParamControls: React.FC<{
  job: WorkflowJob;
  values: Record<string, JobParamValue>;
  onChange: (jobType: string, key: string, value: JobParamValue) => void;
}> = ({ job, values, onChange }) => {
  if (!job.params?.length) return null;

  return (
    <div className="mt-3 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
      {job.params.map((field) => {
        const ariaLabel = `${job.label} ${field.label}`;
        const value = values[field.key];
        if (field.kind === 'checkbox') {
          return (
            <label
              key={field.key}
              className="flex h-9 items-center gap-2 rounded-lg border border-subtle bg-surface/35 px-2.5 text-xs text-secondary-text"
            >
              <input
                type="checkbox"
                aria-label={ariaLabel}
                checked={value === true}
                onChange={(event) => onChange(job.type, field.key, event.currentTarget.checked)}
                className="h-4 w-4 rounded border-subtle accent-cyan"
              />
              <span className="truncate">{field.label}</span>
            </label>
          );
        }

        if (field.kind === 'select') {
          return (
            <label key={field.key} className="flex min-w-0 flex-col gap-1 text-xs text-muted-text">
              <span className="truncate">{field.label}</span>
              <select
                aria-label={ariaLabel}
                value={typeof value === 'string' ? value : ''}
                onChange={(event) => onChange(job.type, field.key, event.currentTarget.value)}
                className={WORKFLOW_INPUT_CLASS}
              >
                <option value="">默认</option>
                {(field.options || []).map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
          );
        }

        return (
          <label key={field.key} className="flex min-w-0 flex-col gap-1 text-xs text-muted-text">
            <span className="truncate">{field.label}</span>
            <input
              type={field.kind}
              min={field.min}
              max={field.max}
              step={field.step}
              aria-label={ariaLabel}
              value={typeof value === 'string' ? value : ''}
              placeholder={field.placeholder}
              onChange={(event) => onChange(job.type, field.key, event.currentTarget.value)}
              className={WORKFLOW_INPUT_CLASS}
            />
          </label>
        );
      })}
    </div>
  );
};

const WorkflowPage: React.FC = () => {
  const { t } = useUiLanguage();
  const [researchStatus, setResearchStatus] = useState<ResearchStatus | null>(null);
  const [quantStatus, setQuantStatus] = useState<QuantStatus | null>(null);
  const [jobs, setJobs] = useState<QuantJob[]>([]);
  const [jobParams, setJobParams] = useState<JobParamState>({});
  const [selectedLogJob, setSelectedLogJob] = useState<QuantJob | null>(null);
  const [selectedArtifactName, setSelectedArtifactName] = useState<string | null>(null);
  const [logContent, setLogContent] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [submittingTypes, setSubmittingTypes] = useState<Record<string, boolean>>({});
  const submittingJobTypesRef = useRef<Set<string>>(new Set());

  const latestJobs = useMemo(() => quantStatus?.latestJobs ?? {}, [quantStatus]);
  const workflowStatus = useMemo<ResearchStatus | null>(() => {
    if (researchStatus) return researchStatus;
    if (!quantStatus) return null;
    return {
      generatedAt: quantStatus.generatedAt,
      trainable: quantStatus.trainable,
      trainingReadiness: quantStatus.trainingReadiness || { status: 'unknown', summary: '' },
      dataSources: quantStatus.dataSources,
      qlibProvider: quantStatus.qlibProvider,
      latestTraining: quantStatus.latestTraining,
      cashControlBacktest: quantStatus.cashControlBacktest,
      experimentNotebook: quantStatus.experimentNotebook,
    };
  }, [researchStatus, quantStatus]);

  const refresh = useCallback(async () => {
    setError(null);
    try {
      const [nextResearchStatus, nextQuantStatus, nextJobs] = await Promise.all([
        researchApi.getStatus(),
        quantApi.getStatus(),
        quantApi.listJobs({ limit: 30 }),
      ]);
      setResearchStatus(nextResearchStatus);
      setQuantStatus(nextQuantStatus);
      setJobs(nextJobs.items);
    } catch (err: unknown) {
      setError(workflowErrorMessage(err));
    } finally {
      setLoading(false);
    }
  }, []);

  const refreshSummary = useCallback(async () => {
    setError(null);
    try {
      const [nextResearchStatus, nextQuantStatus] = await Promise.all([
        researchApi.getStatus(),
        quantApi.getStatus(),
      ]);
      setResearchStatus(nextResearchStatus);
      setQuantStatus(nextQuantStatus);
    } catch (err: unknown) {
      setError(workflowErrorMessage(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    document.title = '量化任务中心 - DSA';
    void refresh();
    const timer = window.setInterval(() => {
      if (typeof window.EventSource === 'undefined') {
        void refresh();
      } else {
        void refreshSummary();
      }
    }, 30000);
    return () => window.clearInterval(timer);
  }, [refresh, refreshSummary]);

  useEffect(() => {
    if (typeof window.EventSource === 'undefined') {
      return undefined;
    }

    let source: EventSource | null = null;
    let reconnectTimer: number | undefined;
    let reconnectDelayMs = STREAM_RECONNECT_INITIAL_DELAY_MS;
    let closed = false;

    const handleJobs = (event: MessageEvent) => {
      const nextJobs = parseQuantJobStreamEvent(event.data);
      setJobs(nextJobs);
      setQuantStatus((current) => current
        ? {
            ...current,
            activeCount: activeJobCount(nextJobs),
            latestJobs: { ...current.latestJobs, ...latestJobsByType(nextJobs) },
            recentJobs: nextJobs,
          }
        : current);
    };

    const handleStreamEnd = () => {
      if (closed) return;
      source?.removeEventListener('jobs', handleJobs as EventListener);
      source?.removeEventListener('stream_end', handleStreamEnd as EventListener);
      source?.close();
      if (reconnectTimer !== undefined) return;
      reconnectTimer = window.setTimeout(() => {
        reconnectTimer = undefined;
        connect();
      }, 0);
    };

    const connect = () => {
      source = new window.EventSource(quantApi.getJobStreamUrl(), { withCredentials: true });
      source.addEventListener('jobs', handleJobs as EventListener);
      source.addEventListener('stream_end', handleStreamEnd as EventListener);
      source.onopen = () => {
        reconnectDelayMs = STREAM_RECONNECT_INITIAL_DELAY_MS;
        setError(null);
      };
      source.onerror = () => {
        if (closed) return;
        setError(t('workflow.streamDisconnected'));
        source?.removeEventListener('jobs', handleJobs as EventListener);
        source?.removeEventListener('stream_end', handleStreamEnd as EventListener);
        source?.close();
        const delay = reconnectDelayMs;
        reconnectDelayMs = Math.min(reconnectDelayMs * 2, STREAM_RECONNECT_MAX_DELAY_MS);
        reconnectTimer = window.setTimeout(() => {
          reconnectTimer = undefined;
          connect();
        }, delay);
      };
    };

    connect();
    return () => {
      closed = true;
      if (reconnectTimer !== undefined) window.clearTimeout(reconnectTimer);
      source?.removeEventListener('jobs', handleJobs as EventListener);
      source?.removeEventListener('stream_end', handleStreamEnd as EventListener);
      source?.close();
    };
  }, [t]);

  const updateJobParam = useCallback((jobType: string, key: string, value: JobParamValue) => {
    setJobParams((current) => ({
      ...current,
      [jobType]: {
        ...(current[jobType] || {}),
        [key]: value,
      },
    }));
  }, []);

  const submitJob = async (workflowJob: WorkflowJob, explicitParams?: Record<string, unknown>) => {
    const jobType = workflowJob.type;
    if (submittingJobTypesRef.current.has(jobType)) {
      return;
    }
    submittingJobTypesRef.current.add(jobType);
    setSubmittingTypes((current) => ({ ...current, [jobType]: true }));
    setError(null);
    try {
      const params = explicitParams ?? buildJobParams(workflowJob, jobParams[jobType]);
      const job = await quantApi.submitJob(jobType, params);
      setJobs((current) => [job, ...current.filter((item) => item.jobId !== job.jobId)]);
      setQuantStatus((current) => current
        ? {
            ...current,
            activeCount: current.activeCount + 1,
            latestJobs: { ...current.latestJobs, [job.jobType]: job },
          }
        : current);
    } catch (err: unknown) {
      setError(workflowErrorMessage(err));
    } finally {
      submittingJobTypesRef.current.delete(jobType);
      setSubmittingTypes((current) => ({ ...current, [jobType]: false }));
    }
  };

  const submitGuideJob = async (jobType: string, params?: Record<string, unknown>) => {
    const workflowJob = WORKFLOW_JOBS_BY_TYPE[jobType];
    if (!workflowJob) {
      setError(`未知任务类型：${jobType}`);
      return;
    }
    await submitJob(workflowJob, params);
  };

  const cancelJob = async (job: QuantJob) => {
    setError(null);
    try {
      const cancelled = await quantApi.cancelJob(job.jobId);
      setJobs((current) => current.map((item) => (item.jobId === cancelled.jobId ? cancelled : item)));
      setQuantStatus((current) => current
        ? {
            ...current,
            activeCount: isActiveJob(job) ? Math.max(0, current.activeCount - 1) : current.activeCount,
            latestJobs: { ...current.latestJobs, [cancelled.jobType]: cancelled },
          }
        : current);
    } catch (err: unknown) {
      setError(workflowErrorMessage(err));
    }
  };

  const openLog = async (job: QuantJob) => {
    setSelectedLogJob(job);
    setSelectedArtifactName(null);
    setLogContent('日志加载中...');
    try {
      const payload = await quantApi.getJobLog(job.jobId);
      setLogContent(payload.content || '日志为空。');
    } catch (err: unknown) {
      setLogContent(workflowErrorMessage(err));
    }
  };

  const previewArtifact = async (job: QuantJob, artifactIndex: number, path: string) => {
    setSelectedLogJob(job);
    setSelectedArtifactName(artifactFileName(path));
    setLogContent('产物加载中...');
    try {
      const content = await quantApi.getJobArtifactText(job.jobId, artifactIndex);
      setLogContent(content || '产物为空。');
    } catch (err: unknown) {
      setLogContent(workflowErrorMessage(err));
    }
  };

  const trainable = workflowStatus?.trainable;
  const latestTraining = workflowStatus?.latestTraining;
  const dailyGuideSteps = useMemo(
    () => buildDailyGuideSteps(workflowStatus, latestJobs),
    [workflowStatus, latestJobs],
  );

  return (
    <main className="min-h-screen bg-base px-4 py-6 sm:px-6 lg:px-8">
      <div className="mx-auto flex max-w-7xl flex-col gap-5">
        <header className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <span className="label-uppercase">Quant Workflow</span>
            <h1 className="mt-2 text-2xl font-semibold tracking-normal text-foreground sm:text-3xl">
              量化任务中心
            </h1>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-secondary-text">
              在网页端完成数据更新、Qlib 转换、质量检查、模型训练、实验追踪和回测查看；后台只执行白名单脚本。
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Link to="/research" className="btn-secondary h-9 gap-2 px-3 text-sm">
              研究总览
              <ArrowRight className="h-4 w-4" />
            </Link>
            <Link to="/backtest" className="btn-secondary h-9 gap-2 px-3 text-sm">
              回测结果
              <ArrowRight className="h-4 w-4" />
            </Link>
            <button type="button" className="btn-primary h-9 gap-2 px-3 text-sm" onClick={() => void refresh()}>
              <RefreshCw className="h-4 w-4" />
              刷新
            </button>
          </div>
        </header>

        {error ? <InlineAlert variant="danger" title="工作流接口异常" message={error} /> : null}

        <section className="grid gap-3 md:grid-cols-2 xl:grid-cols-5">
          <Card variant="bordered" padding="md" className="min-h-[132px]">
            <div className="flex items-center justify-between gap-3">
              <span className="label-uppercase">Data</span>
              <JobStatusBadge job={undefined} />
            </div>
            <p className="mt-3 text-lg font-semibold text-foreground">
              {trainable ? '当前数据可训练' : loading ? '正在读取数据状态' : '数据暂不可训练'}
            </p>
            <p className="mt-2 text-sm leading-5 text-secondary-text">
              {workflowStatus?.trainingReadiness?.summary || '运行质量报告后，这里会显示训练准备状态。'}
            </p>
          </Card>

          <Card variant="bordered" padding="md" className="min-h-[132px]">
            <span className="label-uppercase">Qlib Provider</span>
            <p className="mt-3 text-lg font-semibold text-foreground">
              {workflowStatus?.qlibProvider?.calendarRange || '-'}
            </p>
            <p className="mt-2 text-sm text-secondary-text">
              股票池 {formatNumber(workflowStatus?.qlibProvider?.instruments)} · 日历 {formatNumber(workflowStatus?.qlibProvider?.calendarDays)}
            </p>
          </Card>

          <Card variant="bordered" padding="md" className="min-h-[132px]">
            <span className="label-uppercase">Jobs</span>
            <p className="mt-3 text-lg font-semibold text-foreground">
              {quantStatus?.activeCount ?? 0} 个运行中
            </p>
            <p className="mt-2 text-sm text-secondary-text">
              最近任务 {jobs.length} 条，状态和日志保存在本地 reports/jobs。
            </p>
          </Card>

          <Card variant="bordered" padding="md" className="min-h-[132px]">
            <span className="label-uppercase">Latest Training</span>
            <p className="mt-3 truncate text-lg font-semibold text-foreground">
              {latestTraining?.experimentName || '暂无训练记录'}
            </p>
            <p className="mt-2 truncate text-sm text-secondary-text">
              Run ID: {latestTraining?.runId || '-'}
            </p>
          </Card>

          <Card variant="bordered" padding="md" className="min-h-[132px]">
            <span className="label-uppercase">MLflow</span>
            <p className="mt-3 text-lg font-semibold text-foreground">MLflow 原始记录</p>
            <p className="mt-2 truncate text-sm text-secondary-text">http://127.0.0.1:6006/</p>
            <a
              href="http://127.0.0.1:6006/"
              target="_blank"
              rel="noreferrer"
              aria-label="打开 MLflow，新窗口"
              className="mt-3 inline-flex h-8 items-center gap-1.5 rounded-lg border border-cyan/25 px-2.5 text-xs font-medium text-cyan transition-colors hover:bg-cyan/10"
            >
              打开 MLflow
              <ArrowUpRight className="h-3.5 w-3.5" />
            </a>
          </Card>
        </section>

        <section className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_380px]">
          <div className="flex flex-col gap-4">
            {WORKFLOW_STEPS.map((step) => {
              const Icon = step.icon;
              return (
                <Card key={step.title} variant="bordered" padding="none" className="overflow-hidden">
                  <div className="border-b border-subtle px-4 py-4">
                    <div className="flex items-start gap-3">
                      <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-cyan/20 bg-cyan/10 text-cyan">
                        <Icon className="h-5 w-5" />
                      </div>
                      <div className="min-w-0">
                        <h2 className="text-base font-semibold text-foreground">{step.title}</h2>
                        <p className="mt-1 text-sm leading-5 text-secondary-text">{step.description}</p>
                      </div>
                    </div>
                  </div>
                  <div className="grid gap-0 divide-y divide-subtle">
                    {step.jobs.map((job) => {
                      const latestJob = latestJobs[job.type];
                      const isSubmitting = submittingTypes[job.type] === true;
                      const paramValues = jobParams[job.type] || {};
                      const isKnownJobType = Boolean(WORKFLOW_JOBS_BY_TYPE[job.type]);
                      return (
                        <div key={job.type} className="grid gap-3 px-4 py-3 md:grid-cols-[minmax(0,1fr)_auto] md:items-center">
                          <div className="min-w-0">
                            <div className="flex flex-wrap items-center gap-2">
                              <h3 className="text-sm font-medium text-foreground">{job.label}</h3>
                              <JobStatusBadge job={latestJob} />
                            </div>
                            <p className="mt-1 text-sm leading-5 text-secondary-text">{job.description}</p>
                            <p className="mt-1 text-xs text-muted-text">
                              最近运行：{formatTime(latestJob?.updatedAt)}
                              {latestJob?.error ? ` · ${latestJob.error}` : ''}
                            </p>
                            {latestJob?.artifactPaths?.length ? (
                              <div className="mt-2 flex flex-wrap items-center gap-1.5 text-xs text-muted-text">
                                <span>产物：</span>
                                {latestJob.artifactPaths.slice(0, 4).map((path, artifactIndex) => {
                                  const fileName = artifactFileName(path);
                                  return (
                                    <span key={path} className="inline-flex max-w-full items-center gap-1 rounded-md border border-subtle bg-surface/60 px-1.5 py-0.5">
                                      <a
                                        href={quantApi.getJobArtifactUrl(latestJob.jobId, artifactIndex)}
                                        target="_blank"
                                        rel="noreferrer"
                                        className="max-w-[14rem] truncate text-[11px] text-cyan transition-colors hover:text-cyan/80"
                                      >
                                        {fileName}
                                      </a>
                                      <button
                                        type="button"
                                        className="inline-flex h-5 items-center gap-1 rounded px-1 text-[11px] font-medium text-secondary-text transition-colors hover:bg-cyan/10 hover:text-cyan"
                                        aria-label={`预览 ${fileName}`}
                                        onClick={() => void previewArtifact(latestJob, artifactIndex, path)}
                                      >
                                        <Eye className="h-3 w-3" />
                                        预览
                                      </button>
                                    </span>
                                  );
                                })}
                                {latestJob.artifactPaths.length > 4 ? (
                                  <span>+{latestJob.artifactPaths.length - 4}</span>
                                ) : null}
                              </div>
                            ) : null}
                            {isKnownJobType ? (
                              <WorkflowParamControls
                                job={job}
                                values={paramValues}
                                onChange={updateJobParam}
                              />
                            ) : null}
                          </div>
                          <div className="flex flex-wrap items-center gap-2">
                            {latestJob ? (
                              <button type="button" className="btn-secondary h-9 gap-2 px-3 text-sm" onClick={() => void openLog(latestJob)}>
                                <TerminalSquare className="h-4 w-4" />
                                日志
                              </button>
                            ) : null}
                            {latestJob && (latestJob.status === 'running' || latestJob.status === 'pending') ? (
                              <button
                                type="button"
                                className="btn-secondary h-9 gap-2 px-3 text-sm text-warning"
                                onClick={() => void cancelJob(latestJob)}
                                aria-label={`取消 ${job.label}`}
                              >
                                <CircleStop className="h-4 w-4" />
                                取消
                              </button>
                            ) : null}
                            <button
                              type="button"
                              className="btn-primary h-9 min-w-[8rem] gap-2 px-3 text-sm"
                              onClick={() => void submitJob(job)}
                              disabled={isSubmitting}
                              aria-label={`运行 ${job.label}`}
                            >
                              {isSubmitting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                              运行
                            </button>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </Card>
              );
            })}
          </div>

          <aside className="flex flex-col gap-4">
            <Card variant="bordered" padding="md">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <ListChecks className="h-4 w-4 text-cyan" />
                    <h2 className="text-sm font-semibold text-foreground">今日流程</h2>
                  </div>
                  <p className="mt-1 text-xs leading-5 text-secondary-text">
                    按顺序完成检查、报告、预演、训练和记录。
                  </p>
                </div>
              </div>
              <div className="mt-3 divide-y divide-subtle">
                {dailyGuideSteps.map((step, index) => (
                  <div key={step.id} className="py-3 first:pt-0 last:pb-0">
                    <div className="flex items-start gap-2.5">
                      <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-cyan/25 bg-cyan/10 text-[11px] font-semibold text-cyan">
                        {index + 1}
                      </span>
                      <div className="min-w-0 flex-1">
                        <div className="flex flex-wrap items-center gap-2">
                          <p className="text-sm font-medium text-foreground">{step.title}</p>
                          <Badge variant={step.statusTone === 'success' ? 'success' : step.statusTone === 'danger' ? 'danger' : step.statusTone === 'warning' ? 'warning' : 'info'} size="sm" className="gap-1 shadow-none">
                            <StatusDot tone={step.statusTone} pulse={step.status === 'running'} className="h-1.5 w-1.5" />
                            {step.statusLabel}
                          </Badge>
                        </div>
                        <p className="mt-1 text-xs leading-5 text-secondary-text">{step.detail}</p>
                        {step.meta ? (
                          <p className="mt-1 line-clamp-2 text-[11px] leading-4 text-muted-text">{step.meta}</p>
                        ) : null}
                        <div className="mt-2 flex flex-wrap gap-1.5">
                          {step.actions.map((action) => {
                            if (action.kind === 'job') {
                              const isSubmitting = submittingTypes[action.jobType] === true;
                              return (
                                <button
                                  key={`${step.id}-${action.jobType}-${action.label}`}
                                  type="button"
                                  className="inline-flex h-7 items-center gap-1.5 rounded-lg border border-cyan/25 px-2 text-xs font-medium text-cyan transition-colors hover:bg-cyan/10 disabled:cursor-not-allowed disabled:opacity-60"
                                  onClick={() => void submitGuideJob(action.jobType, action.params)}
                                  disabled={isSubmitting}
                                  aria-label={`今日流程 ${step.title} ${action.label}`}
                                >
                                  {isSubmitting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
                                  {action.label}
                                </button>
                              );
                            }
                            if (action.kind === 'external') {
                              return (
                                <a
                                  key={`${step.id}-${action.href}`}
                                  href={action.href}
                                  target="_blank"
                                  rel="noreferrer"
                                  className="inline-flex h-7 items-center gap-1.5 rounded-lg border border-cyan/25 px-2 text-xs font-medium text-cyan transition-colors hover:bg-cyan/10"
                                >
                                  {action.label}
                                  <ArrowUpRight className="h-3.5 w-3.5" />
                                </a>
                              );
                            }
                            return (
                              <Link
                                key={`${step.id}-${action.to}`}
                                to={action.to}
                                className="inline-flex h-7 items-center gap-1.5 rounded-lg border border-cyan/25 px-2 text-xs font-medium text-cyan transition-colors hover:bg-cyan/10"
                              >
                                {action.label}
                                <ArrowRight className="h-3.5 w-3.5" />
                              </Link>
                            );
                          })}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </Card>

            <Card variant="bordered" padding="md">
              <div className="flex items-center gap-2">
                <Activity className="h-4 w-4 text-cyan" />
                <h2 className="text-sm font-semibold text-foreground">最近任务</h2>
              </div>
              <div className="mt-3 flex flex-col gap-2">
                {jobs.length === 0 ? (
                  <p className="text-sm text-secondary-text">暂无任务。点击左侧按钮后会出现在这里。</p>
                ) : jobs.slice(0, 12).map((job) => (
                  <button
                    type="button"
                    key={job.jobId}
                    className="rounded-xl border border-subtle bg-surface/40 px-3 py-2 text-left transition-colors hover:bg-hover"
                    onClick={() => void openLog(job)}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="min-w-0 truncate text-sm font-medium text-foreground">{job.label}</span>
                      <StatusDot tone={statusTone(job.status)} pulse={job.status === 'running'} className="h-2 w-2" />
                    </div>
                    <div className="mt-1 flex items-center gap-2 text-xs text-muted-text">
                      <History className="h-3.5 w-3.5" />
                      <span className="truncate">{job.jobId}</span>
                    </div>
                  </button>
                ))}
              </div>
            </Card>

            <Card variant="bordered" padding="md">
              <div className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                  <TerminalSquare className="h-4 w-4 text-cyan" />
                  <h2 className="text-sm font-semibold text-foreground">{selectedArtifactName ? '产物预览' : '任务日志'}</h2>
                </div>
                {selectedLogJob ? <JobStatusBadge job={selectedLogJob} /> : null}
              </div>
              {selectedLogJob ? (
                <div className="mt-3">
                  <div className="mb-2 flex items-center gap-2 text-xs text-muted-text">
                    <CheckCircle2 className="h-3.5 w-3.5" />
                    <span className="truncate">{selectedArtifactName || selectedLogJob.jobId}</span>
                  </div>
                  <pre
                    role="log"
                    aria-label={selectedArtifactName ? `产物预览 ${selectedArtifactName}` : `任务日志 ${selectedLogJob.jobId}`}
                    className="max-h-[420px] overflow-auto rounded-xl border border-subtle bg-black/30 p-3 text-xs leading-5 text-secondary-text"
                  >
                    {logContent}
                  </pre>
                </div>
              ) : (
                <p className="mt-3 text-sm text-secondary-text">选择一个最近任务或点击任务行里的“日志”。</p>
              )}
            </Card>
          </aside>
        </section>
      </div>
    </main>
  );
};

export default WorkflowPage;
