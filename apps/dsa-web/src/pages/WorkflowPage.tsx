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
  FlaskConical,
  History,
  ListChecks,
  Loader2,
  Pause,
  Play,
  RefreshCw,
  RotateCw,
  TerminalSquare,
} from 'lucide-react';
import { Badge, Card, ConfirmDialog, InlineAlert, StatusDot } from '../components/common';
import TrainingResultsPanel from '../components/workflow/TrainingResultsPanel';
import { useUiLanguage } from '../contexts/UiLanguageContext';
import { getParsedApiError } from '../api/error';
import {
  parseQuantJobStreamEvent,
  QUANT_API_CONTRACT_VERSION,
  quantApi,
  type QuantApiContract,
  type QuantJob,
  type QuantJobTypeInfo,
  type QuantStatus,
} from '../api/quant';
import { researchApi, type ResearchStatus } from '../api/research';
import { WEB_BUILD_INFO } from '../utils/constants';
import { resolveMlflowUiUrl } from '../utils/mlflowUiUrl';

type WorkflowJob = {
  type: string;
  label: string;
  description: string;
  testOnly?: boolean;
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

export type QuantTaskSurface = 'all' | 'data' | 'training' | 'other';

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
const ROBUST_RUNNER_DISCOVERY_DELAY_MS = 1500;
const ROBUST_RUNNER_POLL_INTERVAL_MS = 3000;
const DURABLE_ROBUST_JOB_TYPES = new Set([
  'run_robust_signal_research',
  'resume_robust_signal_research',
]);
const RECENT_JOB_LIMIT = 30;
const WORKFLOW_INPUT_CLASS =
  'input-surface input-focus-glow h-9 w-full rounded-lg border bg-transparent px-2.5 text-xs transition-all focus:outline-none';

const WORKFLOW_STEPS: WorkflowStep[] = [
  {
    title: '实验输入检查',
    description: '读取已冻结的行情、日历与标的列表，无需连接 QMT。',
    icon: RefreshCw,
    jobs: [
      { type: 'run_daily_update', label: '检查冻结实验输入', description: '检查已选输入能否离线读取；缺失行情请在 QMT 中维护后重新导出。' },
    ],
  },
  {
    title: '独立参考数据',
    description: '保留独立新闻与国际市场参考能力，不参与 QMT 行情合并。',
    icon: Database,
    jobs: [
      {
        type: 'download_global_daily',
        label: '下载全球日线与原始证据',
        description: '从 Yahoo Chart 增量下载美股、日股、韩股日线；每个代码保留一个 canonical parquet，并按抓取日期保存原始响应。',
        defaultParams: { incremental: true },
        params: [
          { key: 'start', label: '开始日期', kind: 'date' },
          { key: 'end', label: '结束日期', kind: 'date' },
          { key: 'limit', label: '代码数量上限', kind: 'number', min: 1 },
          { key: 'sleep', label: '请求间隔秒', kind: 'number', min: 2, step: 0.1, placeholder: '2' },
          { key: 'retries', label: '重试次数', kind: 'number', min: 1, max: 1, placeholder: '1' },
          { key: 'timeout', label: '超时秒数', kind: 'number', min: 1 },
          { key: 'overlap_days', label: '增量重叠天数', kind: 'number', min: 1, max: 365, placeholder: '7' },
          { key: 'incremental', label: '增量下载', kind: 'checkbox' },
          { key: 'use_system_proxy', label: '使用系统代理', kind: 'checkbox' },
        ],
      },
      {
        type: 'download_ashare_news',
        label: '下载 A 股新闻原始证据',
        description: '按限定股票数保存东方财富新闻搜索的原始响应；不允许网页端无上限抓取全市场。',
        params: [
          { key: 'limit', label: '股票数量上限', kind: 'number', min: 1, placeholder: '50' },
          { key: 'sleep', label: '请求间隔秒', kind: 'number', min: 0, step: 0.1, placeholder: '0.1' },
          { key: 'retries', label: '重试次数', kind: 'number', min: 0 },
          { key: 'timeout', label: '超时秒数', kind: 'number', min: 1 },
          { key: 'use_system_proxy', label: '使用系统代理', kind: 'checkbox' },
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
    title: 'Qlib Provider',
    description: '检查研究输入和独立历史研究能力。',
    icon: RotateCw,
    jobs: [
      { type: 'check_qlib_data', label: '检查 Qlib 数据', description: '快速读取样本股票和字段，确认 provider 可用。' },
      {
        type: 'prepare_robust_signal_data',
        label: '准备冻结的鲁棒 CSI800 数据',
        description: '用当前 cleaned Qlib Provider、历史时点 CSI800 成员区间和当前数据血缘构建不可变研究 Provider；数据血缘变化后必须重新运行。',
      },
    ],
  },
  {
    title: '辅助报告',
    description: '查看独立研究的只读辅助报告。',
    icon: ListChecks,
    jobs: [
      {
        type: 'report_workbench_capabilities',
        label: '生成工作台能力盘点',
        description: '只读扫描 DSA 页面/API、quant_workflow 脚本、Qlib 接入和 china-astock-quant 可复用能力，输出功能入口和缺口清单。',
      },
      {
        type: 'generate_etf_momentum_report',
        label: '生成 ETF 完整策略报告',
        description: '读取两只 ETF 的隔离行情，生成收益、回撤、稳定性、持仓和数据质量报告；不连接券商，不自动下单。',
      },
      {
        type: 'run_china_astock_comparison',
        label: '生成 china-astock 策略对比',
        description: '读取 china-astock-quant 既有搜索和赢家评估结果，生成策略对比 CSV/Markdown 报告；只读报告，不连接券商或交易接口。',
      },
      {
        type: 'report_china_astock_indicators',
        label: '生成 china-astock 技术指标报告',
        description: '复用 china-astock-quant 的 SMA/EMA/RSI/ATR/Bollinger 指标函数，读取 Qlib Provider 前复权行情生成只读技术面观察报告，不连接券商、不自动下单。',
        params: [
          { key: 'symbols', label: '股票代码', kind: 'text', placeholder: '000001,600519；留空按数量上限抽样' },
          { key: 'lookback_days', label: '回看交易日', kind: 'number', min: 20, placeholder: '120' },
          { key: 'limit', label: '股票数量上限', kind: 'number', min: 1, max: 500, placeholder: '50' },
        ],
      },
    ],
  },
  {
    title: '选股模型训练与交易策略验证',
    description: '运行单个选股模型基线或五个选股模型候选流程，并记录 MLflow 实验。',
    icon: FlaskConical,
    jobs: [
      {
        type: 'train_baseline_lgb',
        label: '训练 Qlib LightGBM 选股模型基线',
        description: '用冻结 QMT 行情训练 OHLCV 基线，写入独立 MLflow 实验，不覆盖已有模型或 latest。',
        defaultParams: { provider: 'qmt_native' },
        params: [
          {
            key: 'provider',
            label: '数据版本',
            kind: 'select',
            options: [
              { value: 'qmt_native', label: '冻结 QMT 实验输入' },
            ],
          },
          { key: 'config', label: '配置文件', kind: 'text', placeholder: 'baseline_qmt_native.yaml' },
          { key: 'experiment_name', label: '实验名称', kind: 'text', placeholder: 'dsa_qmt_native' },
        ],
      },
      {
        type: 'train_baseline_lgb_equivalence',
        label: '运行真实 LightGBM 网页等价验收',
        description: '真实训练使用服务器固定短周期配置和专用 MLflow 实验；测试结果不会进入正式候选、最新结果或普通回测列表。',
        testOnly: true,
        defaultParams: { provider: 'qmt_native' },
        params: [
          {
            key: 'provider',
            label: '数据版本',
            kind: 'select',
            options: [
              { value: 'qmt_native', label: '冻结 QMT 实验输入' },
            ],
          },
        ],
      },
      {
        type: 'run_candidate_pipeline',
        label: '运行五个选股模型候选流程',
        description: '依次训练 Linear OLS、Linear Ridge、LightGBM、XGBoost 和 CatBoost 五个选股模型；只用 2024-2025 选择候选选股模型及五套交易策略，再用 2026 留出期终审。结果写入 MLflow 和候选流程报告，不自动下单。',
        defaultParams: { provider: 'canonical', dry_run: false },
        params: [
          {
            key: 'provider',
            label: '数据版本',
            kind: 'select',
            options: [
              { value: 'canonical', label: '历史 Provider（canonical）' },
              { value: 'cleaned', label: 'Qlib Provider（cleaned）' },
            ],
          },
          { key: 'dry_run', label: '只做小样本预演', kind: 'checkbox' },
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
        description: '读取最新 Qlib/MLflow pred.pkl，生成最新交易日 Top/Bottom 选股模型评分摘要；只读排行，不连接券商、不自动下单。',
        defaultParams: { topk: '20' },
        params: [
          { key: 'topk', label: 'TopK', kind: 'number', min: 1, max: 500, placeholder: '20' },
          { key: 'run_id', label: 'MLflow Run ID', kind: 'text', placeholder: '留空读取最新训练' },
        ],
      },
      {
        type: 'optimize_strategy',
        label: '交易策略优化预演',
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
              { value: '4', label: '综合交易策略' },
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
  {
    title: 'Qlib研究增强',
    description: '把已有 Qlib/MLflow 产物拆成选股模型解释、交易策略回测明细、选股模型对比、因子、滚动训练和组合风险诊断报告。',
    icon: Activity,
    jobs: [
      {
        type: 'report_qlib_model_explainability',
        label: '生成 Qlib 选股模型解释报告',
        description: '读取最新或指定 run 的 params/pred/label，输出特征重要性、Top/Bottom 预测差异和过拟合诊断；只读报告。',
        defaultParams: { topk: '20' },
        params: [
          { key: 'topk', label: 'TopK', kind: 'number', min: 1, max: 500, placeholder: '20' },
          { key: 'run_id', label: 'MLflow Run ID', kind: 'text', placeholder: '留空读取最新训练' },
        ],
      },
      {
        type: 'report_qlib_backtest_detail',
        label: '生成 Qlib 回测明细报告',
        description: '读取 Qlib portfolio_analysis，输出收益、夏普、回撤、月度收益、持仓和交易活动代理；只读报告。',
        params: [
          { key: 'run_id', label: 'MLflow Run ID', kind: 'text', placeholder: '留空读取最新训练' },
        ],
      },
      {
        type: 'report_qlib_model_comparison',
        label: '生成 Qlib 多选股模型/多实验对比',
        description: '扫描本地 MLflow 最近 runs，对比关键指标和预测相关性；可用于 LightGBM、Linear、CatBoost 等实验横向比较。',
        defaultParams: { max_runs: '8' },
        params: [
          { key: 'max_runs', label: '对比 Run 数', kind: 'number', min: 1, max: 50, placeholder: '8' },
        ],
      },
      {
        type: 'report_qlib_factor_analysis',
        label: '生成 Qlib 因子分析报告',
        description: '基于 score 与 label 计算 IC、RankIC、分组收益和特征重要性代理，作为后续 Alpha158/自定义因子筛选入口。',
        defaultParams: { topk: '20' },
        params: [
          { key: 'topk', label: 'TopK', kind: 'number', min: 1, max: 500, placeholder: '20' },
          { key: 'run_id', label: 'MLflow Run ID', kind: 'text', placeholder: '留空读取最新训练' },
        ],
      },
      {
        type: 'report_qlib_rolling_training',
        label: '生成 Qlib 滚动训练计划',
        description: '读取 Qlib 日历和 baseline 配置，生成月度滚动训练窗口计划；本任务只规划，不启动批量训练。',
        defaultParams: { windows: '6', train_years: '3', valid_months: '12', test_months: '1', step_months: '1' },
        params: [
          { key: 'windows', label: '窗口数', kind: 'number', min: 1, max: 60, placeholder: '6' },
          { key: 'train_years', label: '训练年数', kind: 'number', min: 1, max: 20, placeholder: '3' },
          { key: 'valid_months', label: '验证月数', kind: 'number', min: 1, max: 120, placeholder: '12' },
          { key: 'test_months', label: '测试月数', kind: 'number', min: 1, max: 120, placeholder: '1' },
          { key: 'step_months', label: '滚动步长月', kind: 'number', min: 1, max: 120, placeholder: '1' },
          { key: 'config', label: '配置文件', kind: 'text', placeholder: 'baseline_lgb_alpha158.yaml' },
        ],
      },
      {
        type: 'report_qlib_risk_model',
        label: '生成 Qlib 组合风险诊断',
        description: '从组合收益、benchmark、换手和最新持仓中估算 beta、集中度、权重上限和风险约束缺口；只读报告。',
        params: [
          { key: 'run_id', label: 'MLflow Run ID', kind: 'text', placeholder: '留空读取最新训练' },
        ],
      },
    ],
  },
];

const DATA_JOB_TYPES = new Set([
  'run_daily_update',
  'download_global_daily',
  'download_ashare_news',
  'download_jqdata',
  'check_qlib_data',
  'prepare_robust_signal_data',
]);

const PROVIDER_TRAINING_JOB_TYPES = new Set([
  'train_baseline_lgb',
  'train_baseline_lgb_equivalence',
  'run_candidate_pipeline',
]);

const ROBUST_SIGNAL_JOB_TYPES = new Set([
  'preview_robust_signal_plan',
  'run_robust_signal_research',
  'pause_robust_signal_research',
  'resume_robust_signal_research',
]);

const TRAINING_JOB_TYPES = new Set([
  ...PROVIDER_TRAINING_JOB_TYPES,
  ...ROBUST_SIGNAL_JOB_TYPES,
]);

const ROBUST_SIGNAL_JOBS: WorkflowJob[] = [
  {
    type: 'preview_robust_signal_plan',
    label: '预览 CSI800 选股信号稳定性检验计划',
    description: '只生成任务数量、时间范围和配置指纹，不启动训练。',
  },
  {
    type: 'run_robust_signal_research',
    label: '启动 CSI800 选股信号稳定性检验',
    description: '运行单季度测试或正式鲁棒信号研究。',
  },
  {
    type: 'pause_robust_signal_research',
    label: '完成当前子任务后暂停',
    description: '请求运行器在当前原子任务保存完成后协作式暂停。',
  },
  {
    type: 'resume_robust_signal_research',
    label: '恢复 CSI800 选股信号稳定性检验',
    description: '校验原检查点上下文后恢复同一个研究 Run。',
  },
];

function isTrainingDryRun(jobType: string, params: Record<string, unknown>): boolean {
  return (jobType === 'run_candidate_pipeline' && params.dry_run === true)
    || (jobType === 'run_robust_signal_research' && params.smoke === true);
}

function isFormalTrainingBlocked(
  jobType: string,
  params: Record<string, unknown>,
  status: ResearchStatus | null,
): boolean {
  if (jobType === 'run_robust_signal_research' || jobType === 'resume_robust_signal_research') {
    return status?.robustSignal?.dataReadiness?.trainable !== true
      || status.robustSignal.dataReadiness.status !== 'ready';
  }
  const requiresAcceptance = PROVIDER_TRAINING_JOB_TYPES.has(jobType);
  return requiresAcceptance
    && !isTrainingDryRun(jobType, params)
    && providerTrainingState(status, params).trainable !== true;
}

function selectedProvider(params: Record<string, unknown>): 'qmt_native' | 'canonical' | 'cleaned' {
  return params.provider === 'cleaned' ? 'cleaned' : params.provider === 'canonical' ? 'canonical' : 'qmt_native';
}

function providerTrainingState(
  status: ResearchStatus | null,
  params: Record<string, unknown>,
): {
  provider: 'qmt_native' | 'canonical' | 'cleaned';
  trainable: boolean;
  snapshotId?: string;
  experimentId?: string;
  calendarEnd?: string;
  summary: string;
} {
  const provider = selectedProvider(params);
  const currentIdentity = status?.currentProviderIdentities?.[provider] as (
    NonNullable<ResearchStatus['currentProviderIdentities']>[string]
    & { experimentId?: string; barsSha256?: string }
  ) | undefined;
  const preflight = status?.trainingPreflights?.[provider]
    || (status?.trainingPreflight?.provider === provider ? status.trainingPreflight : undefined);
  const currentSnapshotReady = currentIdentity?.status === 'ready'
    && (provider === 'qmt_native'
      ? Boolean(currentIdentity.experimentId && currentIdentity.barsSha256)
      : Boolean(currentIdentity.snapshotId));
  const trainable = currentSnapshotReady && preflight?.status === 'passed';
  return {
    provider,
    trainable,
    snapshotId: currentSnapshotReady ? currentIdentity?.snapshotId : undefined,
    experimentId: currentSnapshotReady ? currentIdentity?.experimentId : undefined,
    calendarEnd: currentIdentity?.calendarEnd,
    summary: !currentSnapshotReady
      ? currentIdentity?.error || '当前 Qlib Provider 不可用。'
      : preflight?.summary || '当前实时检查状态不可用。',
  };
}

function surfaceForJob(jobType: string): Exclude<QuantTaskSurface, 'all'> {
  if (DATA_JOB_TYPES.has(jobType)) return 'data';
  if (TRAINING_JOB_TYPES.has(jobType)) return 'training';
  return 'other';
}

function stepsForSurface(surface: QuantTaskSurface): WorkflowStep[] {
  if (surface === 'all') return WORKFLOW_STEPS;
  return WORKFLOW_STEPS.flatMap((step) => {
    const jobs = step.jobs.filter((job) => surfaceForJob(job.type) === surface);
    if (!jobs.length) return [];
    if (surface === 'training') {
      return [{
        ...step,
        title: '训练选股模型',
        description: '训练单个选股模型基线或五个选股模型候选流程；训练完成后把选股模型和交易策略组合送入回测页。',
        jobs,
      }];
    }
    if (surface === 'other' && step.title === '辅助报告') {
      return [{ ...step, title: '工作台与外部项目报告', description: '集中查看尚未归入数据、训练和正式回测页面的辅助能力。', jobs }];
    }
    if (surface === 'other' && step.title === '选股模型训练与交易策略验证') {
      return [{ ...step, title: '市场、预测与历史结果', description: '保留市场状态、预测摘要和旧版历史验证任务，等待后续决定。', jobs }];
    }
    return [{ ...step, jobs }];
  });
}

function surfaceCopy(surface: QuantTaskSurface, language: 'zh' | 'en') {
  const zh = language === 'zh';
  if (surface === 'data') {
    return {
      eyebrow: 'Quant Data',
      title: zh ? '数据更新与检查' : 'Data Updates and Checks',
      description: zh
        ? '只读检查冻结实验输入；行情由 QMT 维护。'
        : 'Inspect frozen input; QMT maintains the underlying market data.',
    };
  }
  if (surface === 'training') {
    return {
      eyebrow: 'Model Training',
      title: zh ? '选股模型训练' : 'Stock Selection Model Training',
      description: zh
        ? '训练选股模型，查看候选选股模型与交易策略组合，并把指定结果加载到回测页。'
        : 'Train stock-selection models, inspect model and trading-strategy combinations, and load a result into backtesting.',
    };
  }
  if (surface === 'other') {
    return {
      eyebrow: 'Other Quant Tasks',
      title: zh ? '其他量化任务' : 'Other Quant Tasks',
      description: zh
        ? '集中放置暂未归入数据、选股模型训练或正式回测页面的报告和辅助任务。'
        : 'Reports and supporting tasks not yet assigned to data, model training, or formal backtesting.',
    };
  }
  return {
    eyebrow: 'Quant Workflow',
    title: zh ? '量化任务中心' : 'Quant Workflow',
    description: zh
      ? '查看冻结 QMT 实验输入，运行选股模型训练、实验追踪和 DSA 回测；后台只执行白名单脚本。'
      : 'Inspect frozen QMT input and run training, experiment and DSA backtest tasks through a browser-safe whitelist.',
  };
}

const WORKFLOW_JOBS_BY_TYPE = Object.fromEntries(
  [...WORKFLOW_STEPS.flatMap((step) => step.jobs), ...ROBUST_SIGNAL_JOBS]
    .map((job) => [job.type, job]),
);

function displayQuantJobLabel(job: Pick<QuantJob, 'jobType' | 'label'>): string {
  return WORKFLOW_JOBS_BY_TYPE[job.jobType]?.label || job.label;
}

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

function mlflowStatusMeta(status: string | undefined, language: 'zh' | 'en'): {
  label: string;
  title: string;
  tone: 'success' | 'danger' | 'warning' | 'info' | 'neutral';
  variant: 'success' | 'danger' | 'warning' | 'info' | 'default';
} {
  const labels = language === 'zh'
    ? {
        available: ['可用', 'MLflow 原始记录'],
        linked: ['外部链接', 'MLflow 外部记录'],
        invalid: ['配置无效', 'MLflow 配置待修正'],
        unavailable: ['未启动', 'MLflow 未启动'],
      }
    : {
        available: ['Available', 'MLflow Raw Records'],
        linked: ['External Link', 'MLflow External Records'],
        invalid: ['Invalid Config', 'MLflow Config Needs Fixing'],
        unavailable: ['Unavailable', 'MLflow Unavailable'],
      };
  if (status === 'available') {
    return { label: labels.available[0], title: labels.available[1], tone: 'success', variant: 'success' };
  }
  if (status === 'linked') {
    return { label: labels.linked[0], title: labels.linked[1], tone: 'info', variant: 'info' };
  }
  if (status === 'invalid') {
    return { label: labels.invalid[0], title: labels.invalid[1], tone: 'danger', variant: 'danger' };
  }
  return { label: labels.unavailable[0], title: labels.unavailable[1], tone: 'warning', variant: 'warning' };
}

function serviceStatusMeta(status: string | undefined, language: 'zh' | 'en'): {
  label: string;
  tone: 'success' | 'danger' | 'warning' | 'info' | 'neutral';
  variant: 'success' | 'danger' | 'warning' | 'info' | 'default';
} {
  const labels = language === 'zh'
    ? { available: '可用', linked: '外部链接', invalid: '配置无效', unavailable: '未启动' }
    : { available: 'Available', linked: 'External Link', invalid: 'Invalid Config', unavailable: 'Unavailable' };
  if (status === 'available') return { label: labels.available, tone: 'success', variant: 'success' };
  if (status === 'linked') return { label: labels.linked, tone: 'info', variant: 'info' };
  if (status === 'invalid') return { label: labels.invalid, tone: 'danger', variant: 'danger' };
  return { label: labels.unavailable, tone: 'warning', variant: 'warning' };
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

function formatDuration(seconds?: number | null): string {
  if (seconds === undefined || seconds === null || !Number.isFinite(seconds)) return '-';
  const bounded = Math.max(0, Math.round(seconds));
  if (bounded < 60) return `${bounded}秒`;
  const minutes = Math.floor(bounded / 60);
  if (minutes < 60) return `${minutes}分钟`;
  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;
  return remainingMinutes ? `${hours}小时${remainingMinutes}分钟` : `${hours}小时`;
}

function artifactFileName(path: string): string {
  return path.split(/[\\/]/).filter(Boolean).pop() || path;
}

function parseJobTimestamp(value: string | null | undefined): number {
  const timestamp = new Date(value || '').getTime();
  return Number.isNaN(timestamp) ? 0 : timestamp;
}

function jobTimestamp(job: QuantJob): number {
  return parseJobTimestamp(job.updatedAt || job.createdAt);
}

function isNewerJob(candidate: QuantJob, current: QuantJob): boolean {
  const candidateCreated = parseJobTimestamp(candidate.createdAt);
  const currentCreated = parseJobTimestamp(current.createdAt);
  if (candidateCreated !== currentCreated) return candidateCreated > currentCreated;
  if (candidate.jobId !== current.jobId) return false;
  return parseJobTimestamp(candidate.updatedAt) > parseJobTimestamp(current.updatedAt);
}

function latestJobsByType(jobs: QuantJob[]): Record<string, QuantJob> {
  const latest: Record<string, QuantJob> = {};
  for (const job of jobs) {
    const current = latest[job.jobType];
    if (!current || isNewerJob(job, current)) {
      latest[job.jobType] = job;
    }
  }
  return latest;
}

function mergeLatestJobsByType(
  ...groups: Array<Record<string, QuantJob> | undefined>
): Record<string, QuantJob> {
  return latestJobsByType(groups.flatMap((group) => Object.values(group ?? {})));
}

function jobStatusRank(status: string): number {
  if (status === 'completed' || status === 'failed' || status === 'cancelled') return 2;
  if (status === 'running') return 1;
  return 0;
}

function isNewerJobVersion(candidate: QuantJob, current: QuantJob): boolean {
  const candidateTimestamp = jobTimestamp(candidate);
  const currentTimestamp = jobTimestamp(current);
  if (candidateTimestamp !== currentTimestamp) return candidateTimestamp > currentTimestamp;
  return jobStatusRank(candidate.status) >= jobStatusRank(current.status);
}

function mergeJobsById(...groups: QuantJob[][]): QuantJob[] {
  const merged = new Map<string, QuantJob>();
  for (const job of groups.flat()) {
    const current = merged.get(job.jobId);
    if (!current || isNewerJobVersion(job, current)) {
      merged.set(job.jobId, job);
    }
  }
  const sorted = [...merged.values()]
    .sort((left, right) => jobTimestamp(right) - jobTimestamp(left));
  const activeJobs = sorted.filter(isActiveJob);
  const recentHistory = sorted
    .filter((job) => !isActiveJob(job))
    .slice(0, RECENT_JOB_LIMIT);
  return [...activeJobs, ...recentHistory];
}

function mergeAuthoritativeMonitoringSnapshot(
  currentJobs: QuantJob[],
  incomingJobs: QuantJob[],
): QuantJob[] {
  const activeJobIds = new Set(
    incomingJobs.filter(isActiveJob).map((job) => job.jobId),
  );
  const retainedJobs = currentJobs.filter(
    (job) => !isActiveJob(job) || activeJobIds.has(job.jobId),
  );
  return mergeJobsById(retainedJobs, incomingJobs);
}

function activeJobCount(jobs: QuantJob[]): number {
  return jobs.filter(isActiveJob).length;
}

function isActiveJob(job?: QuantJob): boolean {
  return Boolean(
    job
    && (
      job.canCancel === true
      || job.status === 'pending'
      || job.status === 'running'
    )
  );
}

function isCancelableJob(job?: QuantJob): boolean {
  return job?.canCancel ?? isActiveJob(job);
}

function quantStatusWithJobs(
  status: QuantStatus,
  jobs: QuantJob[],
  previous?: QuantStatus | null,
  authoritativeActive = false,
): QuantStatus {
  const mergedLatestJobs = mergeLatestJobsByType(
    status.latestJobs,
    previous?.latestJobs,
    latestJobsByType(jobs),
  );
  const activeJobIds = new Set(jobs.filter(isActiveJob).map((job) => job.jobId));
  const reconciledLatestJobs = authoritativeActive
    ? Object.fromEntries(
        Object.entries(mergedLatestJobs).filter(
          ([, job]) => !isActiveJob(job) || activeJobIds.has(job.jobId),
        ),
      )
    : mergedLatestJobs;
  return {
    ...status,
    activeCount: activeJobCount(jobs),
    latestJobs: reconciledLatestJobs,
    recentJobs: jobs,
  };
}

function isDoneJob(job?: QuantJob): boolean {
  return job?.status === 'completed' && !isActiveJob(job);
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
  if (isActiveJob(job)) {
    const statusLabel = job.status === 'running' || job.status === 'pending'
      ? '运行中'
      : '后台仍运行';
    return { status: 'running', statusLabel, statusTone: 'info', meta: formatTime(job.updatedAt) };
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
  mlflowUrl: string,
): DailyGuideStep[] {
  const reviewJob = latestJobForTypes(latestJobs, ['report_market_regime', 'report_cash_control_backtest', 'train_baseline_lgb', 'run_dsa_backtest']);

  const dataStatus = researchStatus?.trainable
    ? {
        status: 'done' as const,
        statusLabel: '可训练',
        statusTone: 'success' as const,
        meta: researchStatus.qlibProvider?.calendarRange ? `Qlib: ${researchStatus.qlibProvider.calendarRange}` : undefined,
      }
    : jobDrivenStatus(latestJobs.check_qlib_data, 'todo', '待检查');

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
      detail: '检查冻结行情、日历与标的列表，确认本次实验输入可读取。',
      ...dataStatus,
      actions: [
        { kind: 'job', label: '检查输入', jobType: 'run_daily_update' },
        { kind: 'link', label: '研究总览', to: '/research' },
      ],
    },
    {
      id: 'review',
      title: '训练/复盘',
      detail: '跑 Qlib LightGBM 基线、现金仓位叠加或 DSA 历史准确性回测，然后进入回测页看结果。',
      ...reviewStatus,
      actions: [
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
        { kind: 'external', label: 'MLflow', href: mlflowUrl },
      ],
    },
  ];
}

function workflowErrorMessage(err: unknown): string {
  const parsed = getParsedApiError(err);
  return parsed.message || parsed.rawMessage || '请求未成功完成，请稍后重试。';
}

function backendCompatibilityMessage(
  contract: QuantApiContract | null,
  statusContract: QuantApiContract | undefined,
  loadError: string | null,
): string | null {
  if (loadError) {
    return `后端版本过旧或尚未重启，任务目录接口不可用。请重启 DSA 后端后重试。接口错误：${loadError}`;
  }
  if (!contract?.version) {
    return '后端版本过旧或尚未重启，任务目录没有返回版本契约。请重启 DSA 后端后重试。';
  }
  if (contract.version !== QUANT_API_CONTRACT_VERSION) {
    return `后端版本过旧或尚未重启。网页要求 ${QUANT_API_CONTRACT_VERSION}，后端返回 ${contract.version}。请重启 DSA 后端后重试。`;
  }
  if (statusContract?.version && statusContract.version !== contract.version) {
    return '后端接口返回了不一致的版本契约，可能仍有旧服务残留。请重启 DSA 后端后重试。';
  }
  if (
    statusContract?.backendBuildId
    && contract.backendBuildId
    && statusContract.backendBuildId !== contract.backendBuildId
  ) {
    return '量化状态和任务目录来自不同的后端构建，可能仍有旧服务残留。请重启 DSA 后端后重试。';
  }
  return null;
}

function providerFreshnessText(
  state: ReturnType<typeof providerTrainingState>,
  language: 'zh' | 'en',
): string {
  if (state.provider === 'qmt_native') {
    return `${language === 'zh' ? '冻结输入' : 'Frozen input'}: ${state.trainable
      ? (language === 'zh' ? '离线可用' : 'offline ready')
      : (language === 'zh' ? '未就绪' : 'not ready')}${state.calendarEnd ? ` · ${state.calendarEnd}` : ''}`;
  }
  if (!state.trainable) {
    return language === 'zh'
      ? `实时检查未通过${state.calendarEnd ? ` · Provider 日历截止 ${state.calendarEnd}` : ''}`
      : `live checks failed${state.calendarEnd ? ` · Provider calendar through ${state.calendarEnd}` : ''}`;
  }
  const prefix = language === 'zh' ? '实时检查：通过' : 'Live checks: passed';
  if (!state.calendarEnd) {
    return `${prefix} · ${language === 'zh' ? 'Provider 日历截止日期未知' : 'Provider calendar cutoff unknown'}`;
  }
  return `${prefix} · ${language === 'zh'
    ? `Provider 日历截止 ${state.calendarEnd}`
    : `Provider calendar through ${state.calendarEnd}`}`;
}

function localizedTrainingReadinessSummary(
  summary: string | undefined,
  language: 'zh' | 'en',
): string {
  if (!summary || language === 'en') return summary || '';

  const unavailableProviderMatch = summary.match(
    /^The current (canonical|cleaned) (?:Qlib Provider|native V2 snapshot) is unavailable; formal training is blocked\.$/,
  );
  if (unavailableProviderMatch) {
    return `当前 Qlib Provider（${unavailableProviderMatch[1]}）不可用；正式训练已阻止。`;
  }

  const readyPrefix = 'Qlib provider has enough calendar/instrument coverage for the baseline run.';
  if (!summary.startsWith(readyPrefix)) return summary;

  const ready = 'Qlib 数据快照的交易日和股票覆盖已达到基线训练要求。';
  const warningText = summary.slice(readyPrefix.length).trim().replace(/^Warnings:\s*/, '');
  if (!warningText) return ready;

  const warnings = warningText.split(';').map((item) => {
    const warning = item.trim();
    const nonPositiveVolume = warning.match(/^(\d+) symbols have non-positive volume rows$/);
    if (nonPositiveVolume) return `${nonPositiveVolume[1]} 只股票存在非正成交量记录`;
    const extremeReturn = warning.match(/^(\d+) symbols have absolute daily returns above 50%$/);
    if (extremeReturn) return `${extremeReturn[1]} 只股票存在单日绝对涨跌幅超过 50% 的记录`;
    return warning;
  });
  return `${ready} 风险提示：${warnings.join('；')}。`;
}

function getBackendJobInfo(jobTypes: Record<string, QuantJobTypeInfo> | null, jobType: string): {
  catalogLoaded: boolean;
  info?: QuantJobTypeInfo;
  isEnabled: boolean;
  isDisabled: boolean;
  allowedParamKeys: Set<string> | null;
} {
  const info = jobTypes?.[jobType];
  const catalogLoaded = jobTypes !== null;
  const isEnabled = Boolean(info);
  return {
    catalogLoaded,
    info,
    isEnabled,
    isDisabled: catalogLoaded && !isEnabled,
    allowedParamKeys: info ? new Set(info.allowedParams || []) : null,
  };
}

function isAllowedParam(key: string, allowedParamKeys?: Set<string> | null): boolean {
  return !allowedParamKeys || allowedParamKeys.has(key);
}

function filterJobParams(params: Record<string, unknown>, allowedParamKeys?: Set<string> | null): Record<string, unknown> {
  if (!allowedParamKeys) return params;
  return Object.fromEntries(Object.entries(params).filter(([key]) => allowedParamKeys.has(key)));
}

function buildJobParams(
  job: WorkflowJob,
  values: Record<string, JobParamValue> = {},
  allowedParamKeys?: Set<string> | null,
): Record<string, unknown> {
  const params: Record<string, unknown> = filterJobParams({ ...(job.defaultParams || {}) }, allowedParamKeys);
  for (const field of job.params || []) {
    if (!isAllowedParam(field.key, allowedParamKeys)) continue;
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
  const active = isActiveJob(job);
  const tone = active ? 'info' : statusTone(job.status);
  const label = active && job.status !== 'running' && job.status !== 'pending'
    ? '后台仍运行'
    : (STATUS_LABEL[job.status] || job.status);
  return (
    <Badge variant={tone === 'success' ? 'success' : tone === 'danger' ? 'danger' : tone === 'warning' ? 'warning' : 'info'} size="sm" className="gap-1.5 shadow-none">
      <StatusDot tone={tone} pulse={active} className="h-1.5 w-1.5" />
      {label}
    </Badge>
  );
};

const WorkflowParamControls: React.FC<{
  job: WorkflowJob;
  values: Record<string, JobParamValue>;
  onChange: (jobType: string, key: string, value: JobParamValue) => void;
  allowedParamKeys?: Set<string> | null;
}> = ({ job, values, onChange, allowedParamKeys }) => {
  const fields = (job.params || []).filter((field) => isAllowedParam(field.key, allowedParamKeys));
  if (!fields.length) return null;

  return (
    <div className="mt-3 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
      {fields.map((field) => {
        const ariaLabel = `${job.label} ${field.label}`;
        const explicitValue = values[field.key];
        const defaultValue = job.defaultParams?.[field.key];
        const value = explicitValue !== undefined
          ? explicitValue
          : typeof defaultValue === 'boolean'
            ? defaultValue
            : typeof defaultValue === 'string' || typeof defaultValue === 'number'
              ? String(defaultValue)
              : undefined;
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

export const QuantTaskPage: React.FC<{
  surface?: QuantTaskSurface;
  embedded?: boolean;
}> = ({ surface = 'all', embedded = false }) => {
  const { language, t } = useUiLanguage();
  const [researchStatus, setResearchStatus] = useState<ResearchStatus | null>(null);
  const [quantStatus, setQuantStatus] = useState<QuantStatus | null>(null);
  const [jobTypes, setJobTypes] = useState<Record<string, QuantJobTypeInfo> | null>(null);
  const [jobTypesContract, setJobTypesContract] = useState<QuantApiContract | null>(null);
  const [jobTypesLoadError, setJobTypesLoadError] = useState<string | null>(null);
  const [jobs, setJobs] = useState<QuantJob[]>([]);
  const [jobParams, setJobParams] = useState<JobParamState>({});
  const [selectedLogJobId, setSelectedLogJobId] = useState<string | null>(null);
  const [selectedArtifactName, setSelectedArtifactName] = useState<string | null>(null);
  const [logContent, setLogContent] = useState('');
  const [logTruncated, setLogTruncated] = useState(false);
  const [pendingCancelJobId, setPendingCancelJobId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [submittingTypes, setSubmittingTypes] = useState<Record<string, boolean>>({});
  const submittingJobTypesRef = useRef<Set<string>>(new Set());
  const jobsRef = useRef<QuantJob[]>([]);
  const jobStateRevisionRef = useRef(0);
  const researchStatusRevisionRef = useRef(0);
  const artifactPreviewRevisionRef = useRef(0);

  const applyJobSnapshot = useCallback((
    incomingJobs: QuantJob[],
    options: {
      authoritativeActive?: boolean;
      advancesRevision?: boolean;
    } = {},
  ) => {
    if (options.advancesRevision !== false) {
      jobStateRevisionRef.current += 1;
    }
    const nextJobs = options.authoritativeActive
      ? mergeAuthoritativeMonitoringSnapshot(jobsRef.current, incomingJobs)
      : mergeJobsById(jobsRef.current, incomingJobs);
    jobsRef.current = nextJobs;
    setJobs(nextJobs);
    setQuantStatus((current) => (
      current
        ? quantStatusWithJobs(
            current,
            nextJobs,
            undefined,
            options.authoritativeActive === true,
          )
        : current
    ));
  }, []);

  const applyQuantStatusSnapshot = useCallback((
    incomingStatus: QuantStatus,
    requestRevision: number,
  ) => {
    const incomingJobs = [
      ...Object.values(incomingStatus.latestJobs ?? {}),
      ...(incomingStatus.recentJobs ?? []),
    ];
    const authoritativeActive = requestRevision === jobStateRevisionRef.current;
    const nextJobs = authoritativeActive
      ? mergeAuthoritativeMonitoringSnapshot(jobsRef.current, incomingJobs)
      : jobsRef.current;
    const statusForJobs = authoritativeActive
      ? incomingStatus
      : {
          ...incomingStatus,
          activeCount: activeJobCount(nextJobs),
          latestJobs: {},
          recentJobs: [],
        };
    if (authoritativeActive) {
      jobStateRevisionRef.current += 1;
    }
    jobsRef.current = nextJobs;
    setJobs(nextJobs);
    setQuantStatus((current) => quantStatusWithJobs(
      statusForJobs,
      nextJobs,
      current,
      authoritativeActive,
    ));
  }, []);

  const listedLatestJobs = useMemo(() => latestJobsByType(jobs), [jobs]);
  const latestJobs = useMemo(
    () => mergeLatestJobsByType(quantStatus?.latestJobs, listedLatestJobs),
    [listedLatestJobs, quantStatus],
  );
  const visibleWorkflowSteps = useMemo(() => stepsForSurface(surface), [surface]);
  const visibleJobs = useMemo(
    () => surface === 'all' ? jobs : jobs.filter((job) => surfaceForJob(job.jobType) === surface),
    [jobs, surface],
  );
  const selectedLogJob = useMemo(
    () => jobs.find((job) => job.jobId === selectedLogJobId) ?? null,
    [jobs, selectedLogJobId],
  );
  const pendingCancelJob = useMemo(
    () => jobs.find((job) => job.jobId === pendingCancelJobId) ?? null,
    [jobs, pendingCancelJobId],
  );
  const copy = surfaceCopy(surface, language);
  const externalServices = quantStatus?.externalServices ?? {};
  const mlflow = externalServices.mlflow ?? quantStatus?.mlflow;
  const dsaService = externalServices.dsa ?? {
    url: '/',
    status: quantStatus ? 'available' : 'unavailable',
    role: 'local_web_api',
  };
  const catdeskService = externalServices.catdesk ?? {
    url: 'http://127.0.0.1:8765/',
    status: 'unavailable',
    role: 'external_catalyst_dashboard',
  };
  const mlflowUrl = resolveMlflowUiUrl(mlflow?.url || 'http://127.0.0.1:6006/');
  const catdeskUrl = catdeskService.url || 'http://127.0.0.1:8765/';
  const mlflowMeta = mlflowStatusMeta(mlflow?.status, language);
  const externalServiceRows = [
    {
      key: 'dsa',
      label: 'DSA',
      title: language === 'zh' ? '本地研究工作台' : 'Local Research Console',
      url: dsaService.url || '/',
      service: dsaService,
      internal: true,
    },
    {
      key: 'mlflow',
      label: 'MLflow',
      title: mlflowMeta.title,
      url: mlflowUrl,
      service: mlflow,
      internal: false,
    },
    {
      key: 'catdesk',
      label: 'CatDesk',
      title: language === 'zh' ? '美股催化观察台' : 'US Catalyst Desk',
      url: catdeskUrl,
      service: catdeskService,
      internal: false,
    },
  ];
  const workflowStatus = useMemo<ResearchStatus | null>(() => {
    if (researchStatus) return researchStatus;
    if (!quantStatus) return null;
    return {
      generatedAt: quantStatus.generatedAt,
      trainable: quantStatus.trainable,
      trainingReadiness: quantStatus.trainingReadiness || { status: 'unknown', summary: '' },
      trainingPreflight: quantStatus.trainingPreflight,
      trainingPreflights: quantStatus.trainingPreflights,
      dataSources: quantStatus.dataSources,
      qlibProvider: quantStatus.qlibProvider,
      providerQuality: quantStatus.providerQuality,
      currentProviderIdentities: quantStatus.currentProviderIdentities,
      latestTraining: quantStatus.latestTraining,
      cashControlBacktest: quantStatus.cashControlBacktest,
      qlibPredictions: quantStatus.qlibPredictions,
      qlibResearch: quantStatus.qlibResearch,
      robustStrategy: quantStatus.robustStrategy,
      robustSignal: quantStatus.robustSignal,
      experimentNotebook: quantStatus.experimentNotebook,
    };
  }, [researchStatus, quantStatus]);
  const backendCompatibilityIssue = loading
    ? null
    : backendCompatibilityMessage(jobTypesContract, quantStatus?.apiContract, jobTypesLoadError);
  const compatibleJobTypes = backendCompatibilityIssue ? {} : jobTypes;

  const requestResearchStatus = useCallback(async () => {
    const requestRevision = researchStatusRevisionRef.current + 1;
    researchStatusRevisionRef.current = requestRevision;
    try {
      const nextStatus = await researchApi.getStatus();
      if (requestRevision === researchStatusRevisionRef.current) {
        setResearchStatus(nextStatus);
      }
    } catch (err: unknown) {
      if (requestRevision === researchStatusRevisionRef.current) {
        setResearchStatus(null);
        throw err;
      }
    }
  }, []);

  const refresh = useCallback(async () => {
    setError(null);
    setJobTypesLoadError(null);
    const statusRequestRevision = jobStateRevisionRef.current;
    try {
      await Promise.all([
        quantApi.getJobTypes()
          .then((catalog) => {
            setJobTypes(catalog.availableJobTypes);
            setJobTypesContract(catalog.apiContract);
          })
          .catch((err: unknown) => {
            setJobTypes({});
            setJobTypesContract(null);
            setJobTypesLoadError(workflowErrorMessage(err));
          }),
        requestResearchStatus(),
        quantApi.getStatus({ includeResearch: false }).then((nextStatus) => {
          applyQuantStatusSnapshot(nextStatus, statusRequestRevision);
        }),
        quantApi.listJobs({ limit: RECENT_JOB_LIMIT }).then((nextJobs) => {
          applyJobSnapshot(
            nextJobs.items.filter((job) => !isActiveJob(job)),
            { advancesRevision: false },
          );
        }),
      ]);
    } catch (err: unknown) {
      setError(workflowErrorMessage(err));
    } finally {
      setLoading(false);
    }
  }, [applyJobSnapshot, applyQuantStatusSnapshot, requestResearchStatus]);

  const refreshSummary = useCallback(async () => {
    setError(null);
    const statusRequestRevision = jobStateRevisionRef.current;
    try {
      const [, nextQuantStatus] = await Promise.all([
        requestResearchStatus(),
        quantApi.getStatus({ includeResearch: false }),
      ]);
      applyQuantStatusSnapshot(nextQuantStatus, statusRequestRevision);
    } catch (err: unknown) {
      setError(workflowErrorMessage(err));
    } finally {
      setLoading(false);
    }
  }, [applyQuantStatusSnapshot, requestResearchStatus]);

  const refreshResearchStatus = useCallback(async () => {
    try {
      await requestResearchStatus();
    } catch (err: unknown) {
      setError(workflowErrorMessage(err));
    }
  }, [requestResearchStatus]);

  useEffect(() => {
    document.title = `${copy.title} - DSA`;
    void refresh();
    const timer = window.setInterval(() => {
      if (typeof window.EventSource === 'undefined') {
        void refresh();
      } else {
        void refreshSummary();
      }
    }, 30000);
    return () => window.clearInterval(timer);
  }, [copy.title, refresh, refreshSummary]);

  useEffect(() => {
    if (typeof window.EventSource === 'undefined') {
      return undefined;
    }

    let source: EventSource | null = null;
    let reconnectTimer: number | undefined;
    let runnerRefreshTimer: number | undefined;
    let reconnectDelayMs = STREAM_RECONNECT_INITIAL_DELAY_MS;
    let closed = false;

    const handleJobs = (event: MessageEvent) => {
      const nextJobs = parseQuantJobStreamEvent(event.data);
      const previousById = new Map(jobsRef.current.map((job) => [job.jobId, job]));
      const reachedTerminalState = nextJobs.some((job) => {
        const previous = previousById.get(job.jobId);
        return Boolean(previous && isActiveJob(previous) && !isActiveJob(job));
      });
      const durableRunnerLaunched = nextJobs.some((job) => {
        const previous = previousById.get(job.jobId);
        return Boolean(
          previous
          && isActiveJob(previous)
          && job.status === 'completed'
          && DURABLE_ROBUST_JOB_TYPES.has(job.jobType),
        );
      });
      applyJobSnapshot(nextJobs, { authoritativeActive: true });
      if (reachedTerminalState) {
        void refreshResearchStatus();
      }
      if (durableRunnerLaunched) {
        if (runnerRefreshTimer !== undefined) window.clearTimeout(runnerRefreshTimer);
        runnerRefreshTimer = window.setTimeout(() => {
          runnerRefreshTimer = undefined;
          void refreshResearchStatus();
        }, ROBUST_RUNNER_DISCOVERY_DELAY_MS);
      }
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
      if (closed) return;
      source = new window.EventSource(quantApi.getJobStreamUrl(), { withCredentials: true });
      source.addEventListener('jobs', handleJobs as EventListener);
      source.addEventListener('stream_end', handleStreamEnd as EventListener);
      source.onopen = () => {
        reconnectDelayMs = STREAM_RECONNECT_INITIAL_DELAY_MS;
        setError(null);
      };
      source.onerror = () => {
        if (closed) return;
        if (reconnectTimer !== undefined) return;
        setError(t('workflow.streamDisconnected'));
        source?.removeEventListener('jobs', handleJobs as EventListener);
        source?.removeEventListener('stream_end', handleStreamEnd as EventListener);
        source?.close();
        const delay = reconnectDelayMs;
        reconnectDelayMs = Math.min(reconnectDelayMs * 2, STREAM_RECONNECT_MAX_DELAY_MS);
        reconnectTimer = window.setTimeout(() => {
          reconnectTimer = undefined;
          if (closed) return;
          connect();
        }, delay);
      };
    };

    connect();
    return () => {
      closed = true;
      if (reconnectTimer !== undefined) window.clearTimeout(reconnectTimer);
      if (runnerRefreshTimer !== undefined) window.clearTimeout(runnerRefreshTimer);
      source?.removeEventListener('jobs', handleJobs as EventListener);
      source?.removeEventListener('stream_end', handleStreamEnd as EventListener);
      source?.close();
    };
  }, [applyJobSnapshot, refreshResearchStatus, t]);

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
    const backendJobInfo = getBackendJobInfo(compatibleJobTypes, jobType);
    if (!backendJobInfo.catalogLoaded) {
      setError('任务白名单仍在加载，请稍后再试。');
      return;
    }
    if (!backendJobInfo.isEnabled) {
      setError(backendCompatibilityIssue || `后端未启用任务：${workflowJob.label}`);
      return;
    }
    const params = explicitParams
      ? filterJobParams(explicitParams, backendJobInfo.allowedParamKeys)
      : buildJobParams(workflowJob, jobParams[jobType], backendJobInfo.allowedParamKeys);
    if (isFormalTrainingBlocked(jobType, params, workflowStatus)) {
      const providerState = providerTrainingState(workflowStatus, params);
      setError(
        `${providerState.provider} 数据没有通过训练检查：${localizedTrainingReadinessSummary(providerState.summary, language)}`,
      );
      return;
    }
    if (submittingJobTypesRef.current.has(jobType)) {
      return;
    }
    submittingJobTypesRef.current.add(jobType);
    setSubmittingTypes((current) => ({ ...current, [jobType]: true }));
    setError(null);
    try {
      const job = await quantApi.submitJob(jobType, params);
      applyJobSnapshot([job]);
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
      applyJobSnapshot([cancelled]);
      if (cancelled.status !== 'cancelled' && cancelled.error) {
        setError(cancelled.error);
      }
    } catch (err: unknown) {
      setError(workflowErrorMessage(err));
    }
  };

  const openLog = (job: QuantJob) => {
    artifactPreviewRevisionRef.current += 1;
    setSelectedLogJobId(job.jobId);
    setSelectedArtifactName(null);
    setLogContent('日志加载中...');
    setLogTruncated(false);
  };

  const previewArtifact = async (job: QuantJob, artifactIndex: number, path: string) => {
    const requestRevision = artifactPreviewRevisionRef.current + 1;
    artifactPreviewRevisionRef.current = requestRevision;
    setSelectedLogJobId(job.jobId);
    setSelectedArtifactName(artifactFileName(path));
    setLogContent('产物加载中...');
    setLogTruncated(false);
    try {
      const content = await quantApi.getJobArtifactText(job.jobId, artifactIndex);
      if (requestRevision !== artifactPreviewRevisionRef.current) return;
      setLogContent(content || '产物为空。');
    } catch (err: unknown) {
      if (requestRevision !== artifactPreviewRevisionRef.current) return;
      setLogContent(workflowErrorMessage(err));
    }
  };

  useEffect(() => {
    if (!selectedLogJobId || selectedArtifactName) return undefined;
    let active = true;
    void quantApi.getJobLog(selectedLogJobId)
      .then((payload) => {
        if (!active) return;
        setLogContent(payload.content || '日志为空。');
        setLogTruncated(payload.truncated === true);
      })
      .catch((err: unknown) => {
        if (!active) return;
        setLogContent(workflowErrorMessage(err));
        setLogTruncated(false);
      });
    return () => {
      active = false;
    };
  }, [selectedArtifactName, selectedLogJob?.logUpdatedAt, selectedLogJob?.updatedAt, selectedLogJobId]);

  const trainable = workflowStatus?.trainable;
  const latestTraining = workflowStatus?.latestTraining;
  const candidatePipeline = workflowStatus?.candidatePipeline;
  const robustSignal = workflowStatus?.robustSignal;
  const robustDataReadiness = robustSignal?.dataReadiness;
  const robustDataReady = robustDataReadiness?.status === 'ready'
    && robustDataReadiness.trainable === true;
  const robustRunId = robustSignal?.runId;
  const robustStage = robustSignal?.currentStage
    || robustSignal?.progress?.currentStage
    || '-';
  const robustPauseState = robustSignal?.pauseState
    || robustSignal?.progress?.pauseState
    || '';
  const robustLifecycle = robustSignal?.lifecycleStatus || robustSignal?.status || '';
  const robustCompleted = robustSignal?.completedCount
    ?? robustSignal?.progress?.completedCount
    ?? 0;
  const robustExpected = robustSignal?.expectedCount
    ?? robustSignal?.progress?.expectedCount
    ?? 0;
  const robustSmokeRun = robustSignal?.smokeRun;
  const robustSmokeLifecycle = robustSmokeRun?.lifecycleStatus || robustSmokeRun?.status || '';
  const robustSmokePauseState = robustSmokeRun?.pauseState
    || robustSmokeRun?.progress?.pauseState
    || '';
  const robustSmokePaused = robustSmokePauseState === 'paused'
    || robustSmokeLifecycle === 'paused';
  const robustSmokeRunning = robustSmokeLifecycle === 'running'
    || robustSmokePauseState === 'running'
    || robustSmokePauseState === 'pause_requested';
  const robustSmokeCompleted = robustSmokeRun?.completedCount
    ?? robustSmokeRun?.progress?.completedCount
    ?? 0;
  const robustSmokeExpected = robustSmokeRun?.expectedCount
    ?? robustSmokeRun?.progress?.expectedCount
    ?? 0;
  const robustSmokeStage = robustSmokeRun?.currentStage
    || robustSmokeRun?.progress?.currentStage
    || '-';
  const robustSmokeStatusLabel = robustSmokePaused
    ? (language === 'zh' ? '已暂停' : 'paused')
    : robustSmokeRunning
      ? (language === 'zh' ? '运行中' : 'running')
      : robustSmokeLifecycle === 'completed'
      ? (language === 'zh' ? '已完成' : 'completed')
      : robustSmokeLifecycle === 'failed'
        ? (language === 'zh' ? '失败' : 'failed')
        : robustSmokeLifecycle || (language === 'zh' ? '未运行' : 'not run');
  const robustPaused = robustPauseState === 'paused' || robustLifecycle === 'paused';
  const robustRunning = robustLifecycle === 'running'
    || robustPauseState === 'running'
    || robustPauseState === 'pause_requested';
  const robustActiveLauncherJob = useMemo(() => [
    ...jobs,
    latestJobs.run_robust_signal_research,
    latestJobs.resume_robust_signal_research,
  ]
    .filter((job): job is QuantJob => Boolean(
      job
      && DURABLE_ROBUST_JOB_TYPES.has(job.jobType)
      && isActiveJob(job)
    ))
    .sort((left, right) => jobTimestamp(right) - jobTimestamp(left))[0] ?? null, [
    jobs,
    latestJobs.resume_robust_signal_research,
    latestJobs.run_robust_signal_research,
  ]);
  const robustLaunchActive = Boolean(robustActiveLauncherJob);
  const robustStatusLabel = robustPaused
    ? (language === 'zh' ? '已暂停' : 'paused')
    : robustRunning
      ? (robustPauseState === 'pause_requested'
          ? (language === 'zh' ? '等待当前子任务完成后暂停' : 'pause requested')
          : (language === 'zh' ? '运行中' : 'running'))
      : robustLaunchActive
        ? (language === 'zh' ? '后台仍运行' : 'runner active')
      : robustLifecycle || (language === 'zh' ? '未启动' : 'not started');
  const robustPauseTarget = useMemo(() => (
    robustRunning && robustRunId
      ? { runId: robustRunId, smoke: false }
      : robustSmokeRunning && robustSmokeRun?.runId
        ? { runId: robustSmokeRun.runId, smoke: true }
        : robustActiveLauncherJob
          ? {
              runId: String(
                robustActiveLauncherJob.params.run_id
                ?? robustActiveLauncherJob.params.runId
                ?? '',
              ),
              smoke: robustActiveLauncherJob.params.smoke === true,
            }
          : null
  ), [robustActiveLauncherJob, robustRunId, robustRunning, robustSmokeRun?.runId, robustSmokeRunning]);
  const robustCancelableJob = useMemo(() => {
    if (!robustPauseTarget) return null;
    const candidates = [
      ...jobs,
      latestJobs.run_robust_signal_research,
      latestJobs.resume_robust_signal_research,
    ].filter((job): job is QuantJob => Boolean(job));
    return candidates
      .filter((job) => {
        if (
          job.jobType !== 'run_robust_signal_research'
          && job.jobType !== 'resume_robust_signal_research'
        ) return false;
        const runId = job.params.run_id ?? job.params.runId;
        return String(runId || '') === robustPauseTarget.runId && isCancelableJob(job);
      })
      .sort((left, right) => jobTimestamp(right) - jobTimestamp(left))[0] ?? null;
  }, [jobs, latestJobs.resume_robust_signal_research, latestJobs.run_robust_signal_research, robustPauseTarget]);
  const robustResumeTarget = robustPaused && robustRunId
    ? { runId: robustRunId, smoke: false }
    : robustSmokePaused && robustSmokeRun?.runId
      ? { runId: robustSmokeRun.runId, smoke: true }
      : null;
  useEffect(() => {
    if (!robustRunning && !robustSmokeRunning && !robustLaunchActive) return undefined;
    const timer = window.setInterval(() => {
      void refreshResearchStatus();
    }, ROBUST_RUNNER_POLL_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [refreshResearchStatus, robustLaunchActive, robustRunning, robustSmokeRunning]);
  const robustPlanActive = isActiveJob(latestJobs.preview_robust_signal_plan);
  const robustPauseJobActive = isActiveJob(latestJobs.pause_robust_signal_research);
  const robustFormalBlocked = isFormalTrainingBlocked(
    'run_robust_signal_research',
    {},
    workflowStatus,
  );
  const robustPlanJobInfo = getBackendJobInfo(compatibleJobTypes, 'preview_robust_signal_plan');
  const robustRunJobInfo = getBackendJobInfo(compatibleJobTypes, 'run_robust_signal_research');
  const robustPauseJobInfo = getBackendJobInfo(compatibleJobTypes, 'pause_robust_signal_research');
  const robustResumeJobInfo = getBackendJobInfo(compatibleJobTypes, 'resume_robust_signal_research');
  const robustBackendCatalogLoaded = [
    robustPlanJobInfo,
    robustRunJobInfo,
    robustPauseJobInfo,
    robustResumeJobInfo,
  ].every((info) => info.catalogLoaded);
  const robustUnavailableActions = [
    { label: language === 'zh' ? '预览计划' : 'preview plan', info: robustPlanJobInfo },
    { label: language === 'zh' ? '启动研究' : 'start research', info: robustRunJobInfo },
    { label: language === 'zh' ? '协作暂停' : 'cooperative pause', info: robustPauseJobInfo },
    { label: language === 'zh' ? '恢复研究' : 'resume research', info: robustResumeJobInfo },
  ].filter(({ info }) => info.catalogLoaded && !info.isEnabled);
  const providerStates = Object.keys(researchStatus?.currentProviderIdentities || {})
    .filter((provider) => Boolean(
      workflowStatus?.currentProviderIdentities?.[provider]
      || workflowStatus?.trainingPreflights?.[provider],
    ))
    .map((provider) => providerTrainingState(workflowStatus, { provider }));
  const allProvidersTrainable = providerStates.length
    ? providerStates.every((providerState) => providerState.trainable)
    : trainable;
  const hasTrainableProvider = providerStates.length
    ? providerStates.some((providerState) => providerState.trainable)
    : trainable;
  const hasNonTrainableProvider = providerStates.some(
    (providerState) => !providerState.trainable,
  );
  const hasMixedProviderReadiness = hasTrainableProvider && hasNonTrainableProvider;
  const readyProviderNames = providerStates
    .filter((providerState) => providerState.trainable)
    .map((providerState) => providerState.provider);
  const blockedProviderNames = providerStates
    .filter((providerState) => !providerState.trainable)
    .map((providerState) => providerState.provider);
  const mixedReadinessSummary = hasMixedProviderReadiness
    ? language === 'zh'
      ? `${readyProviderNames.join('、')} 实时检查通过；${blockedProviderNames.join('、')} 当前不能用于正式训练。请在对应训练任务中查看原因。`
      : `${readyProviderNames.join(', ')} passed live checks; ${blockedProviderNames.join(', ')} cannot currently run formal training. See the selected training task for details.`
    : null;
  const preflightLabel = loading
    ? (language === 'zh' ? '读取中' : 'loading')
    : allProvidersTrainable
      ? (language === 'zh' ? '通过' : 'passed')
      : hasMixedProviderReadiness
        ? (language === 'zh' ? '部分通过，部分阻塞' : 'partly passed, partly blocked')
        : (language === 'zh' ? '阻塞' : 'blocked');
  const dailyGuideSteps = useMemo(
    () => buildDailyGuideSteps(workflowStatus, latestJobs, mlflowUrl),
    [workflowStatus, latestJobs, mlflowUrl],
  );
  return (
    <main className="min-h-screen bg-base px-4 py-6 sm:px-6 lg:px-8">
      <div className="mx-auto flex max-w-7xl flex-col gap-5">
        {!embedded ? <header className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <span className="label-uppercase">{copy.eyebrow}</span>
            <h1 className="mt-2 text-2xl font-semibold tracking-normal text-foreground sm:text-3xl">
              {copy.title}
            </h1>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-secondary-text">
              {copy.description}
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
        </header> : null}

        {!embedded && surface !== 'all' ? (
          <nav className="flex flex-wrap border-b border-border/70" aria-label={language === 'zh' ? '量化任务分类' : 'Quant task sections'}>
            {[
              { to: '/data', label: language === 'zh' ? '数据更新与检查' : 'Data' , key: 'data' },
              { to: '/workflow', label: language === 'zh' ? '选股模型训练' : 'Model training', key: 'training' },
              { to: '/other', label: language === 'zh' ? '其他量化任务' : 'Other', key: 'other' },
            ].map((item) => (
              <Link
                key={item.key}
                to={item.to}
                aria-current={surface === item.key ? 'page' : undefined}
                className={`inline-flex h-10 items-center border-b-2 px-4 text-sm font-medium transition-colors ${surface === item.key ? 'border-cyan text-cyan' : 'border-transparent text-secondary-text hover:text-foreground'}`}
              >
                {item.label}
              </Link>
            ))}
          </nav>
        ) : null}

        {error ? <InlineAlert variant="danger" title="工作流接口异常" message={error} /> : null}

        <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-text" aria-label={language === 'zh' ? '构建信息' : 'Build information'}>
          <span>{language === 'zh' ? '前端构建' : 'Frontend build'} {WEB_BUILD_INFO.buildId}</span>
          <span>{language === 'zh' ? '后端构建' : 'Backend build'} {jobTypesContract?.backendBuildId || quantStatus?.apiContract?.backendBuildId || '-'}</span>
          <span>{language === 'zh' ? '接口契约' : 'API contract'} {jobTypesContract?.version || '-'}</span>
          <span>{language === 'zh' ? '后端启动' : 'Backend started'} {jobTypesContract?.backendStartedAt || '-'}</span>
        </div>

        {backendCompatibilityIssue ? (
          <InlineAlert
            variant="warning"
            title={language === 'zh' ? '网页与后端版本不兼容' : 'Frontend and backend are incompatible'}
            message={backendCompatibilityIssue}
          />
        ) : null}

        <section className={`grid gap-3 md:grid-cols-2 ${surface === 'training' ? 'xl:grid-cols-2' : 'xl:grid-cols-3'}`}>
          {surface === 'all' || surface === 'data' || surface === 'training' ? (
          <Card variant="bordered" padding="md" className="min-h-[132px]">
            <span className="label-uppercase">Data</span>
            <p className="mt-3 text-lg font-semibold text-foreground">
              {allProvidersTrainable
                ? (language === 'zh' ? '实时检查通过' : 'Live checks passed')
                : loading
                  ? (language === 'zh' ? '正在读取数据状态' : 'Loading data status')
                  : hasMixedProviderReadiness
                    ? (language === 'zh' ? '部分 Provider 实时检查通过' : 'Some Providers passed live checks')
                    : (language === 'zh' ? '数据暂不可训练' : 'Data not ready for training')}
            </p>
            <p className="mt-2 text-sm leading-5 text-secondary-text">
              {mixedReadinessSummary
                || localizedTrainingReadinessSummary(workflowStatus?.trainingReadiness?.summary, language)
                || (language === 'zh'
                  ? '正在检查 quant_data 与当前 Qlib Provider。'
                  : 'Checking quant_data and the current Qlib Provider.')}
            </p>
            <p className={`mt-3 text-xs ${loading ? 'text-muted-text' : allProvidersTrainable ? 'text-success' : 'text-warning'}`}>
              {language === 'zh' ? '实时检查' : 'Live checks'}：{preflightLabel}
            </p>
            {surface === 'training' ? (
              <div className="mt-3 flex flex-wrap items-center gap-2">
                <Link to="/data" className="inline-flex h-8 items-center gap-1 text-xs font-medium text-cyan">
                  {language === 'zh' ? '查看数据页' : 'Open data page'}
                  <ArrowRight className="h-3.5 w-3.5" />
                </Link>
              </div>
            ) : null}
          </Card>
          ) : null}

          {surface === 'all' || surface === 'data' || surface === 'training' ? (
          <Card variant="bordered" padding="md" className="min-h-[132px]">
            <span className="label-uppercase">Qlib Provider</span>
            <p className="mt-3 text-lg font-semibold text-foreground">
              {workflowStatus?.qlibProvider?.calendarRange || '-'}
            </p>
            <p className="mt-2 text-sm text-secondary-text">
              {language === 'zh' ? '股票池' : 'Universe'} {formatNumber(workflowStatus?.qlibProvider?.instruments)} · {language === 'zh' ? '日历' : 'Calendar'} {formatNumber(workflowStatus?.qlibProvider?.calendarDays)}
            </p>
          </Card>
          ) : null}

          {surface === 'all' || surface === 'data' || surface === 'other' ? (
          <Card variant="bordered" padding="md" className="min-h-[132px]">
            <span className="label-uppercase">Jobs</span>
            <p className="mt-3 text-lg font-semibold text-foreground">
              {language === 'zh'
                ? `${activeJobCount(visibleJobs)} 个运行中`
                : `${activeJobCount(visibleJobs)} running`}
            </p>
            <p className="mt-2 text-sm text-secondary-text">
              {language === 'zh'
                ? `当前分类最近任务 ${visibleJobs.length} 条，状态和日志保存在本地。`
                : `${visibleJobs.length} recent jobs in this section; status and logs are stored locally.`}
            </p>
          </Card>
          ) : null}

          {surface === 'all' ? (
          <Card variant="bordered" padding="md" className="min-h-[132px]">
            <div className="flex items-start justify-between gap-3">
              <span className="label-uppercase">Services</span>
              <Badge variant={serviceStatusMeta(catdeskService.status, language).variant} size="sm" className="gap-1.5 shadow-none">
                <StatusDot tone={serviceStatusMeta(catdeskService.status, language).tone} className="h-1.5 w-1.5" />
                {language === 'zh' ? '外部服务' : 'External'}
              </Badge>
            </div>
            <p className="mt-3 text-lg font-semibold text-foreground">{language === 'zh' ? '外部服务状态' : 'External service status'}</p>
            <div className="mt-3 space-y-2">
              {externalServiceRows.map((item) => {
                const meta = serviceStatusMeta(item.service?.status, language);
                const error = item.service?.error;
                const row = (
                  <span className="min-w-0">
                    <span className="block truncate text-sm font-medium text-foreground">{item.label}</span>
                    <span className="block truncate text-xs text-secondary-text">{item.title}</span>
                  </span>
                );
                return (
                  <div key={item.key} className="rounded-lg border border-subtle px-2.5 py-2">
                    <div className="flex items-center justify-between gap-2">
                      {item.internal ? (
                        <Link to={item.url} className="min-w-0 hover:text-cyan">
                          {row}
                        </Link>
                      ) : (
                        <a
                          href={item.url}
                          target="_blank"
                          rel="noreferrer"
                          aria-label={language === 'zh' ? `打开 ${item.label}，新窗口` : `Open ${item.label} in a new window`}
                          className="min-w-0 hover:text-cyan"
                        >
                          {row}
                        </a>
                      )}
                      <Badge variant={meta.variant} size="sm" className="shrink-0 gap-1.5 shadow-none">
                        <StatusDot tone={meta.tone} className="h-1.5 w-1.5" />
                        {meta.label}
                      </Badge>
                    </div>
                    {error ? <p className="mt-1 line-clamp-2 text-xs leading-4 text-warning">{error}</p> : null}
                  </div>
                );
              })}
            </div>
          </Card>
          ) : null}

          {surface === 'other' ? (
          <Card variant="bordered" padding="md" className="min-h-[132px]">
            <span className="label-uppercase">Pending Classification</span>
            <p className="mt-3 text-lg font-semibold text-foreground">
              {visibleWorkflowSteps.reduce((total, step) => total + step.jobs.length, 0)} {language === 'zh' ? '个待分类任务' : 'tasks awaiting classification'}
            </p>
            <p className="mt-2 text-sm leading-5 text-secondary-text">
              {language === 'zh'
                ? '这些能力暂时集中在本页，确认用途后再分别归入训练、回测或研究页面。'
                : 'These capabilities remain here until they are assigned to training, backtesting, or research.'}
            </p>
          </Card>
          ) : null}
        </section>

        {surface === 'training' && !loading && trainable === false ? (
          <section className="flex flex-wrap items-center justify-between gap-3 border border-warning/35 bg-warning/5 px-4 py-3">
            <div>
              <p className="text-sm font-semibold text-warning">
                {language === 'zh' ? '正式训练已锁定' : 'Formal training is locked'}
              </p>
              <p className="mt-1 text-xs leading-5 text-secondary-text">
                {language === 'zh'
                  ? '原因见上方数据状态。LightGBM 选股模型基线和五个选股模型正式训练不会启动；勾选“五个选股模型小样本预演”后只运行内置测试数据。'
                  : 'See the data status above. Formal baseline and five-model training cannot start; the explicit dry-run uses built-in test data only.'}
              </p>
            </div>
            <Link to="/data" className="btn-secondary h-9 gap-2 px-3 text-sm">
              {language === 'zh' ? '去数据页修复' : 'Open data page'}
              <ArrowRight className="h-4 w-4" />
            </Link>
          </section>
        ) : null}

        {surface === 'training' ? (
          <Card variant="bordered" padding="none" className="overflow-hidden">
            <div className="border-b border-subtle px-4 py-4">
              <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-2">
                    <FlaskConical className="h-5 w-5 text-cyan" />
                    <h2 className="text-base font-semibold text-foreground">
                      {language === 'zh' ? 'CSI800 选股信号稳定性检验' : 'CSI800 Stock-Selection Signal Stability Test'}
                    </h2>
                    <Badge
                      variant={robustRunning || robustLaunchActive ? 'info' : robustPaused ? 'warning' : 'default'}
                      size="sm"
                      className="shadow-none"
                    >
                      {robustStatusLabel}
                    </Badge>
                    {!robustBackendCatalogLoaded ? (
                      <Badge variant="default" size="sm" className="shadow-none">
                        {language === 'zh' ? '任务目录加载中' : 'Loading task catalog'}
                      </Badge>
                    ) : robustUnavailableActions.length > 0 ? (
                      <Badge variant="warning" size="sm" className="shadow-none">
                        {backendCompatibilityIssue
                          ? (language === 'zh' ? '后端需重启' : 'Backend restart required')
                          : (language === 'zh' ? '后端未启用鲁棒信号研究任务' : 'Backend robust-signal tasks unavailable')}
                      </Badge>
                    ) : null}
                  </div>
                  <p className="mt-2 max-w-3xl text-sm leading-5 text-secondary-text">
                    {language === 'zh'
                      ? '先预览固定研究矩阵；单季度测试不进入正式结果，正式运行必须使用已验收数据。暂停会在当前原子任务保存后生效。'
                      : 'Preview the fixed matrix first. The one-quarter smoke run stays outside formal results; formal runs require accepted data. Cooperative pause takes effect after the current atomic task is saved.'}
                  </p>
                </div>
                <div className="shrink-0 text-xs leading-5 text-secondary-text lg:text-right">
                  <p>{robustCompleted} / {robustExpected} · {robustStage} · {robustStatusLabel}</p>
                  <p>{language === 'zh' ? '检查点' : 'Checkpoint'}：{robustRunId || '-'}</p>
                </div>
              </div>
            </div>
            <div className="grid gap-4 px-4 py-4 lg:grid-cols-[minmax(0,1fr)_auto] lg:items-end">
                <div className="min-w-0 space-y-1 text-xs leading-5 text-secondary-text">
                {robustSmokeRun ? (
                  <div
                    className="mb-3 border-l-2 border-warning bg-warning/5 px-3 py-2"
                    aria-label={language === 'zh' ? '最近鲁棒信号单季度测试' : 'Latest robust signal one-quarter test'}
                  >
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="font-semibold text-foreground">
                        {language === 'zh' ? '最近单季度测试' : 'Latest one-quarter test'}
                      </span>
                      <Badge variant="warning" size="sm" className="shadow-none">
                        {language === 'zh'
                          ? '测试数据，不可用于研究或交易'
                          : 'Test data, not for research or trading'}
                      </Badge>
                    </div>
                    <p className="mt-1">
                      {robustSmokeCompleted} / {robustSmokeExpected} · {robustSmokeStage} · {robustSmokeStatusLabel}
                    </p>
                    <p>
                      {language === 'zh' ? '测试检查点' : 'Test checkpoint'}：{robustSmokeRun.runId || '-'}
                    </p>
                    {robustSmokeRun.progress?.currentJob ? (
                      <p className="break-all font-mono text-[11px]">
                        {language === 'zh' ? '当前子任务' : 'Current task'}：{robustSmokeRun.progress.currentJob}
                      </p>
                    ) : null}
                  </div>
                ) : null}
                <p className={robustDataReady ? 'text-success' : 'text-warning'}>
                  {language === 'zh' ? '鲁棒 CSI800 数据' : 'Robust CSI800 data'}：
                  {robustDataReady
                    ? (language === 'zh' ? '已就绪' : 'ready')
                    : (language === 'zh' ? '未就绪' : 'blocked')}
                </p>
                <p>
                  {language === 'zh'
                    ? (robustDataReadiness?.summaryZh || '尚未找到鲁棒 CSI800 专用数据验收记录。')
                    : (robustDataReadiness?.summary || 'No dedicated robust CSI800 readiness record was found.')}
                </p>
                {robustDataReadiness?.providerKey ? (
                  <p>
                    {language === 'zh' ? '鲁棒数据快照' : 'Robust data snapshot'} {robustDataReadiness.providerKey}
                  </p>
                ) : null}
                <p>
                  {language === 'zh' ? '配置' : 'Config'} {robustSignal?.configHash || '-'}
                  {' · '}{language === 'zh' ? '数据' : 'Data'} {robustSignal?.dataManifestHash || '-'}
                  {' · '}{language === 'zh' ? '代码' : 'Code'} {robustSignal?.gitCommit || '-'}
                </p>
                {robustSignal?.contextHash ? (
                  <p>{language === 'zh' ? '完整上下文' : 'Full context'} {robustSignal.contextHash}</p>
                ) : null}
                {robustSignal?.nextGate ? (
                  <p>{language === 'zh' ? '下一门槛' : 'Next gate'}：{robustSignal.nextGate}</p>
                ) : null}
                {robustUnavailableActions.length > 0 ? (
                  <p className="text-warning">
                    {language === 'zh' ? '不可用操作：' : 'Unavailable actions: '}
                    {robustUnavailableActions.map(({ label }) => label).join(language === 'zh' ? '、' : ', ')}
                    {language === 'zh' ? '。' : '.'}
                  </p>
                ) : null}
              </div>
              <div className="flex flex-wrap items-center gap-2">
                {!robustDataReady ? (
                  <Link
                    to="/data"
                    className="btn-secondary h-9 gap-2 px-3 text-sm"
                    aria-label={language === 'zh' ? '去数据页准备鲁棒 CSI800 数据' : 'Open data page to prepare robust CSI800 data'}
                  >
                    <Database className="h-4 w-4" />
                    {language === 'zh' ? '准备鲁棒数据' : 'Prepare data'}
                  </Link>
                ) : null}
                <button
                  type="button"
                  className="btn-secondary h-9 gap-2 px-3 text-sm"
                  aria-label={language === 'zh' ? '预览鲁棒信号研究计划' : 'Preview robust signal plan'}
                  onClick={() => void submitGuideJob('preview_robust_signal_plan', {})}
                  disabled={
                    !robustPlanJobInfo.catalogLoaded
                    || !robustPlanJobInfo.isEnabled
                    || robustPlanActive
                    || submittingTypes.preview_robust_signal_plan === true
                  }
                >
                  {robustPlanActive || submittingTypes.preview_robust_signal_plan === true
                    ? <Loader2 className="h-4 w-4 animate-spin" />
                    : <Eye className="h-4 w-4" />}
                  {language === 'zh' ? '预览计划' : 'Preview plan'}
                </button>
                <button
                  type="button"
                  className="btn-secondary h-9 gap-2 px-3 text-sm"
                  aria-label={language === 'zh' ? '运行鲁棒信号单季度测试' : 'Run one-quarter robust signal smoke test'}
                  onClick={() => void submitGuideJob('run_robust_signal_research', { smoke: true })}
                  disabled={
                    !robustRunJobInfo.catalogLoaded
                    || !robustRunJobInfo.isEnabled
                    || !robustDataReady
                    || robustLaunchActive
                    || robustRunning
                    || robustSmokeRunning
                    || submittingTypes.run_robust_signal_research === true
                  }
                >
                  <FlaskConical className="h-4 w-4" />
                  {language === 'zh' ? '单季度测试' : 'One-quarter test'}
                </button>
                <button
                  type="button"
                  className="btn-primary h-9 gap-2 px-3 text-sm"
                  aria-label={language === 'zh' ? '正式启动鲁棒信号研究' : 'Start formal robust signal research'}
                  onClick={() => void submitGuideJob('run_robust_signal_research', {})}
                  disabled={
                    !robustRunJobInfo.catalogLoaded
                    || !robustRunJobInfo.isEnabled
                    || robustFormalBlocked
                    || robustLaunchActive
                    || robustRunning
                    || robustSmokeRunning
                    || submittingTypes.run_robust_signal_research === true
                  }
                >
                  <Play className="h-4 w-4" />
                  {language === 'zh' ? '正式启动' : 'Start formal run'}
                </button>
                <button
                  type="button"
                  className="btn-secondary h-9 gap-2 px-3 text-sm text-warning"
                  aria-label={language === 'zh' ? '完成当前子任务后暂停' : 'Pause after current task'}
                  onClick={() => robustPauseTarget && void submitGuideJob(
                    'pause_robust_signal_research',
                    { run_id: robustPauseTarget.runId },
                  )}
                  disabled={
                    !robustPauseJobInfo.catalogLoaded
                    || !robustPauseJobInfo.isEnabled
                    || !robustPauseTarget
                    || robustPauseJobActive
                    || submittingTypes.pause_robust_signal_research === true
                  }
                >
                  <Pause className="h-4 w-4" />
                  {language === 'zh' ? '协作暂停' : 'Cooperative pause'}
                </button>
                {robustPauseTarget ? (
                  <button
                    type="button"
                    className="btn-secondary h-9 gap-2 px-3 text-sm text-warning"
                    aria-label={language === 'zh'
                      ? '取消正在运行的鲁棒信号研究'
                      : 'Cancel the running robust signal research'}
                    onClick={() => robustCancelableJob && setPendingCancelJobId(robustCancelableJob.jobId)}
                    disabled={!robustCancelableJob}
                  >
                    <CircleStop className="h-4 w-4" />
                    {language === 'zh' ? '立即取消' : 'Cancel now'}
                  </button>
                ) : null}
                <button
                  type="button"
                  className="btn-secondary h-9 gap-2 px-3 text-sm"
                  aria-label={language === 'zh' ? '恢复鲁棒信号研究' : 'Resume robust signal research'}
                  onClick={() => robustResumeTarget && void submitGuideJob(
                    'resume_robust_signal_research',
                    {
                      run_id: robustResumeTarget.runId,
                      ...(robustResumeTarget.smoke ? { smoke: true } : {}),
                    },
                  )}
                  disabled={
                    !robustResumeJobInfo.catalogLoaded
                    || !robustResumeJobInfo.isEnabled
                    || !robustResumeTarget
                    || robustLaunchActive
                    || (!robustResumeTarget.smoke && robustFormalBlocked)
                    || submittingTypes.resume_robust_signal_research === true
                  }
                >
                  <RotateCw className="h-4 w-4" />
                  {language === 'zh' ? '恢复' : 'Resume'}
                </button>
              </div>
            </div>
          </Card>
        ) : null}

        <section className="grid min-w-0 grid-cols-[minmax(0,1fr)] gap-4 xl:grid-cols-[minmax(0,1fr)_380px]">
          <div className="flex flex-col gap-4">
            {visibleWorkflowSteps.map((step) => {
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
                      const jobIsActive = Boolean(latestJob && isActiveJob(latestJob));
                      const isSubmitting = submittingTypes[job.type] === true;
                      const paramValues = jobParams[job.type] || {};
                      const isKnownJobType = Boolean(WORKFLOW_JOBS_BY_TYPE[job.type]);
                      const backendJobInfo = getBackendJobInfo(compatibleJobTypes, job.type);
                      const effectiveParams = buildJobParams(job, paramValues, backendJobInfo.allowedParamKeys);
                      const providerState = providerTrainingState(workflowStatus, effectiveParams);
                      const formalTrainingBlocked = isFormalTrainingBlocked(job.type, effectiveParams, workflowStatus);
                      return (
                        <div
                          key={job.type}
                          data-job-type={job.type}
                          data-testid={`workflow-job-${job.type}`}
                          className="grid gap-3 px-4 py-3 md:grid-cols-[minmax(0,1fr)_auto] md:items-center"
                        >
                          <div className="min-w-0">
                            <div className="flex flex-wrap items-center gap-2">
                              <h3 className="text-sm font-medium text-foreground">{job.label}</h3>
                              {job.testOnly ? (
                                <Badge variant="warning" size="sm" className="shadow-none">
                                  {language === 'zh' ? '隔离测试，不发布' : 'Isolated test only'}
                                </Badge>
                              ) : null}
                              <JobStatusBadge job={latestJob} />
                              {backendJobInfo.isDisabled ? (
                                <Badge variant="warning" size="sm" className="shadow-none">
                                  {backendCompatibilityIssue ? '后端需重启' : '后端未启用'}
                                </Badge>
                              ) : null}
                              {formalTrainingBlocked && !loading ? (
                                <Badge variant="warning" size="sm" className="shadow-none">
                                  {language === 'zh' ? '等待实验输入' : 'Waiting for data'}
                                </Badge>
                              ) : null}
                            </div>
                            <p className="mt-1 text-sm leading-5 text-secondary-text">{job.description}</p>
                            {TRAINING_JOB_TYPES.has(job.type) ? (
                              <p className={`mt-1 text-xs ${providerState.trainable ? 'text-success' : 'text-warning'}`}>
                                {language === 'zh' ? '当前数据' : 'Current data'}：{providerState.provider}
                                {' · '}{providerState.provider === 'qmt_native'
                                  ? (language === 'zh' ? '实验输入' : 'Experiment input')
                                  : (language === 'zh' ? '历史快照' : 'Historical snapshot')}
                                {' '}{providerState.experimentId || providerState.snapshotId || '-'}
                                {' · '}{providerFreshnessText(providerState, language)}
                                {!providerState.trainable
                                  ? ` · ${localizedTrainingReadinessSummary(providerState.summary, language)}`
                                  : ''}
                              </p>
                            ) : null}
                            <p className="mt-1 text-xs text-muted-text">
                              最近运行：{formatTime(latestJob?.updatedAt)}
                              {latestJob?.error ? ` · ${latestJob.error}` : ''}
                            </p>
                            {latestJob?.progress && (
                              (latestJob.progress.total ?? 0) > 0
                              || Boolean(latestJob.progress.stage)
                            ) ? (
                              <div className="mt-2 max-w-xl space-y-1.5 rounded-md border border-subtle bg-surface/45 px-2.5 py-2 text-xs text-secondary-text">
                                <div className="flex flex-wrap items-center justify-between gap-2">
                                  <span>
                                    {latestJob.progress.completed ?? 0} / {latestJob.progress.total ?? 0}
                                    {' · '}{Number(latestJob.progress.percent ?? 0).toLocaleString('zh-CN', { maximumFractionDigits: 2 })}%
                                  </span>
                                  <span>
                                    阶段：{latestJob.progress.stage || '-'}
                                    {latestJob.progress.current ? ` · 当前：${latestJob.progress.current}` : ''}
                                  </span>
                                </div>
                                <div className="h-1.5 overflow-hidden rounded-full bg-border/45" aria-hidden="true">
                                  <div
                                    className="h-full bg-cyan transition-[width] duration-300"
                                    style={{ width: `${Math.min(100, Math.max(0, Number(latestJob.progress.percent ?? 0)))}%` }}
                                  />
                                </div>
                                <p>
                                  已用 {formatDuration(latestJob.progress.elapsedSeconds)}
                                  {latestJob.progress.etaSeconds !== null && latestJob.progress.etaSeconds !== undefined
                                    ? ` · 预计剩余 ${formatDuration(latestJob.progress.etaSeconds)}`
                                    : ''}
                                  {latestJob.progress.currentArtifact
                                    ? ` · 当前产物 ${latestJob.progress.currentArtifact}`
                                    : ''}
                                </p>
                                {latestJob.pid ? (
                                  <p className={latestJob.runnerAlive === false && isActiveJob(latestJob) ? 'text-warning' : 'text-muted-text'}>
                                    Runner PID {latestJob.pid} · {latestJob.runnerAlive ? '存活' : '未存活'}
                                  </p>
                                ) : null}
                              </div>
                            ) : null}
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
                            {isKnownJobType && backendJobInfo.isEnabled ? (
                              <WorkflowParamControls
                                job={job}
                                values={paramValues}
                                onChange={updateJobParam}
                                allowedParamKeys={backendJobInfo.allowedParamKeys}
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
                            {latestJob && isCancelableJob(latestJob) ? (
                              <button
                                type="button"
                                className="btn-secondary h-9 gap-2 px-3 text-sm text-warning"
                                onClick={() => setPendingCancelJobId(latestJob.jobId)}
                                aria-label={`取消 ${job.label}`}
                              >
                                <CircleStop className="h-4 w-4" />
                                取消
                              </button>
                            ) : null}
                            <button
                              type="button"
                              data-testid={`workflow-run-${job.type}`}
                              className="btn-primary h-9 min-w-[8rem] gap-2 px-3 text-sm"
                              onClick={() => void submitJob(job)}
                              disabled={jobIsActive || isSubmitting || !backendJobInfo.catalogLoaded || !backendJobInfo.isEnabled || formalTrainingBlocked}
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

          {surface === 'training' ? (
            <div className="xl:col-span-2 xl:row-start-2">
              <TrainingResultsPanel
                latestTraining={latestTraining}
                candidatePipeline={candidatePipeline}
                language={language}
                mlflowUrl={mlflowUrl}
              />
            </div>
          ) : null}

          <aside className={`flex flex-col gap-4 ${surface === 'training' ? 'xl:col-start-2 xl:row-start-1' : ''}`}>
            {surface === 'all' ? <Card variant="bordered" padding="md">
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
                              const backendJobInfo = getBackendJobInfo(compatibleJobTypes, action.jobType);
                              return (
                                <span key={`${step.id}-${action.jobType}-${action.label}`} className="inline-flex items-center gap-1.5">
                                  <button
                                    type="button"
                                    className="inline-flex h-7 items-center gap-1.5 rounded-lg border border-cyan/25 px-2 text-xs font-medium text-cyan transition-colors hover:bg-cyan/10 disabled:cursor-not-allowed disabled:opacity-60"
                                    onClick={() => void submitGuideJob(action.jobType, action.params)}
                                    disabled={isSubmitting || isActiveJob(latestJobs[action.jobType]) || !backendJobInfo.catalogLoaded || !backendJobInfo.isEnabled}
                                    aria-label={`今日流程 ${step.title} ${action.label}`}
                                  >
                                    {isSubmitting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
                                    {action.label}
                                  </button>
                                  {backendJobInfo.isDisabled ? (
                                    <Badge variant="warning" size="sm" className="shadow-none">
                                      后端未启用
                                    </Badge>
                                  ) : null}
                                </span>
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
            </Card> : null}

            <Card variant="bordered" padding="md">
              <div className="flex items-center gap-2">
                <Activity className="h-4 w-4 text-cyan" />
                <h2 className="text-sm font-semibold text-foreground">最近任务</h2>
              </div>
              <div className="mt-3 flex flex-col gap-2">
                {visibleJobs.length === 0 ? (
                  <p className="text-sm text-secondary-text">暂无任务。运行当前页面的任务后会出现在这里。</p>
                ) : visibleJobs.slice(0, 12).map((job) => (
                  <button
                    type="button"
                    key={job.jobId}
                    className="rounded-xl border border-subtle bg-surface/40 px-3 py-2 text-left transition-colors hover:bg-hover"
                    onClick={() => void openLog(job)}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="min-w-0 truncate text-sm font-medium text-foreground">{displayQuantJobLabel(job)}</span>
                      <StatusDot
                        tone={isActiveJob(job) ? 'info' : statusTone(job.status)}
                        pulse={isActiveJob(job)}
                        className="h-2 w-2"
                      />
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
                  {logTruncated && !selectedArtifactName ? (
                    <p className="mt-2 text-xs text-warning">日志过长，仅显示最后一部分。</p>
                  ) : null}
                </div>
              ) : (
                <p className="mt-3 text-sm text-secondary-text">选择一个最近任务或点击任务行里的“日志”。</p>
              )}
            </Card>
          </aside>
        </section>
      </div>
      <ConfirmDialog
        isOpen={pendingCancelJob !== null}
        title={language === 'zh' ? '确认取消任务' : 'Cancel this job?'}
        message={language === 'zh'
          ? `取消会立即终止“${pendingCancelJob ? displayQuantJobLabel(pendingCancelJob) : ''}”。这不是可恢复的暂停，已完成的检查点由具体训练流程自行保留。`
          : `Cancelling stops “${pendingCancelJob ? displayQuantJobLabel(pendingCancelJob) : ''}” immediately. This is not a resumable pause.`}
        confirmText={language === 'zh' ? '确认取消' : 'Cancel job'}
        cancelText={language === 'zh' ? '继续运行' : 'Keep running'}
        isDanger
        onCancel={() => setPendingCancelJobId(null)}
        onConfirm={() => {
          const job = pendingCancelJob;
          setPendingCancelJobId(null);
          if (job) void cancelJob(job);
        }}
      />
    </main>
  );
};

const WorkflowPage: React.FC = () => <QuantTaskPage surface="training" />;

export default WorkflowPage;
