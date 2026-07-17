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
  ShieldCheck,
  TerminalSquare,
} from 'lucide-react';
import { Badge, Card, InlineAlert, StatusDot } from '../components/common';
import { useUiLanguage } from '../contexts/UiLanguageContext';
import { getParsedApiError } from '../api/error';
import { parseQuantJobStreamEvent, quantApi, type QuantJob, type QuantJobTypeInfo, type QuantStatus } from '../api/quant';
import { researchApi, type ResearchStatus } from '../api/research';
import { resolveMlflowUiUrl } from '../utils/mlflowUiUrl';

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
    title: '一键全链路',
    description: '把下载、合并、清洗和交叉验证串成一次自动运行；无新交易日时自动跳过重建。',
    icon: RefreshCw,
    jobs: [
      {
        type: 'run_daily_update',
        label: '运行每日数据更新全链路',
        description:
          '按顺序执行：Baostock/东方财富/AkShare 增量下载 → 合并转换 Qlib → 新浪快照校验 → 异常检测 → 清洗 → 多源交叉验证 → provider 校验。' +
          'Baostock 失败会让日更失败；东方财富/AkShare 失败会记录为警告并继续后续可执行阶段；JQData 只参与合并与交叉验证，不在此下载。',
        params: [
          { key: 'skip_download', label: '跳过下载只重建', kind: 'checkbox' },
          { key: 'force', label: '强制重建（无新数据也跑）', kind: 'checkbox' },
          { key: 'dry_run', label: '仅预演不执行', kind: 'checkbox' },
          { key: 'continue_on_error', label: '出错继续后续阶段', kind: 'checkbox' },
          { key: 'skip_eastmoney', label: '跳过东方财富', kind: 'checkbox' },
          { key: 'skip_akshare', label: '跳过 AkShare', kind: 'checkbox' },
          { key: 'skip_snapshot_validation', label: '跳过快照校验', kind: 'checkbox' },
          { key: 'skip_robust_shadow', label: '跳过鲁棒策略纸面跟踪', kind: 'checkbox' },
          { key: 'use_system_proxy', label: '使用系统代理', kind: 'checkbox' },
          { key: 'end', label: '结束日期', kind: 'date' },
          { key: 'overlap_days', label: '增量重叠天数', kind: 'number', min: 1, max: 365, placeholder: '45' },
          { key: 'sleep', label: '请求间隔秒', kind: 'number', min: 0, step: 0.1, placeholder: '0.3' },
          { key: 'timeout', label: '超时秒数', kind: 'number', min: 1, placeholder: '60' },
          { key: 'retries', label: '重试次数', kind: 'number', min: 0, placeholder: '3' },
          { key: 'max_consecutive_failures', label: '连续失败上限', kind: 'number', min: 1, placeholder: '50' },
          { key: 'limit', label: '股票数量上限', kind: 'number', min: 1, placeholder: '留空为全部' },
          { key: 'snapshot_limit', label: '快照校验样本', kind: 'number', min: 1, placeholder: '50' },
          { key: 'overlap_sample', label: '交叉验证重叠样本', kind: 'number', min: 1, placeholder: '留空为全部' },
          { key: 'merged_sample', label: '质量报告合并样本', kind: 'number', min: 1, placeholder: '留空为全部' },
        ],
      },
    ],
  },
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
        type: 'download_eastmoney',
        label: '下载东方财富日线',
        description: '匿名抓取东财 qfq 日线，写入 eastmoney_qfq 作为 Baostock/JQData 的冗余源。',
        defaultParams: { incremental: true },
        params: [
          { key: 'start', label: '开始日期', kind: 'date' },
          { key: 'end', label: '结束日期', kind: 'date' },
          { key: 'limit', label: '股票数量上限', kind: 'number', min: 1 },
          { key: 'sleep', label: '请求间隔秒', kind: 'number', min: 0, step: 0.1, placeholder: '0.1' },
          { key: 'retries', label: '重试次数', kind: 'number', min: 0 },
          { key: 'timeout', label: '超时秒数', kind: 'number', min: 1 },
          { key: 'max_consecutive_failures', label: '连续失败上限', kind: 'number', min: 1 },
          { key: 'overlap_days', label: '增量重叠天数', kind: 'number', min: 1, max: 365, placeholder: '30' },
          { key: 'incremental', label: '增量下载', kind: 'checkbox' },
          { key: 'retry_failed', label: '只补未缓存', kind: 'checkbox' },
          { key: 'use_system_proxy', label: '使用系统代理', kind: 'checkbox' },
        ],
      },
      {
        type: 'download_akshare',
        label: '下载 AkShare 日线',
        description: '调用 AkShare stock_zh_a_hist(qfq)，写入 akshare_qfq 作为第二条免费冗余源。',
        defaultParams: { incremental: true },
        params: [
          { key: 'start', label: '开始日期', kind: 'date' },
          { key: 'end', label: '结束日期', kind: 'date' },
          { key: 'limit', label: '股票数量上限', kind: 'number', min: 1 },
          { key: 'sleep', label: '请求间隔秒', kind: 'number', min: 0, step: 0.1, placeholder: '0.2' },
          { key: 'retries', label: '重试次数', kind: 'number', min: 0 },
          { key: 'timeout', label: '超时秒数', kind: 'number', min: 1 },
          { key: 'max_consecutive_failures', label: '连续失败上限', kind: 'number', min: 1 },
          { key: 'overlap_days', label: '增量重叠天数', kind: 'number', min: 1, max: 365, placeholder: '30' },
          { key: 'incremental', label: '增量下载', kind: 'checkbox' },
          { key: 'retry_failed', label: '只补未缓存', kind: 'checkbox' },
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
        label: '预演修复 Qlib Provider',
        description: '检查日历、股票池、benchmark 等 provider 元数据修复计划；网页端固定 dry-run，不写入 provider。',
      },
      {
        type: 'generate_csi300',
        label: '预演生成 CSI300 股票池',
        description: '读取当前股票池和 Qlib 日历，预演生成 csi300.txt；网页端固定 dry-run，不写入 provider。',
      },
      {
        type: 'add_index_benchmark',
        label: '预演添加沪深300 benchmark',
        description: '检查当前 provider 日历和 benchmark 目标目录；网页端固定 dry-run，不请求 Baostock、不写 bin、不改股票池。',
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
        type: 'validate_latest_snapshot',
        label: '校验新浪最新快照',
        description: '抽样读取新浪行情快照，对比合并后的最新收盘价；只生成校验报告，不参与训练。',
        params: [
          { key: 'limit', label: '校验样本数', kind: 'number', min: 1, placeholder: '50' },
          { key: 'timeout', label: '超时秒数', kind: 'number', min: 1, placeholder: '10' },
          { key: 'use_system_proxy', label: '使用系统代理', kind: 'checkbox' },
        ],
      },
      {
        type: 'report_data_hygiene',
        label: '生成数据治理盘点和清理预案',
        description: '只读盘点当前数据、日志、旧 provider 和临时脚本，生成 inventory 与 dry-run 清理计划；不移动、不删除、不改配置。',
      },
      {
        type: 'report_workbench_capabilities',
        label: '生成工作台能力盘点',
        description: '只读扫描 DSA 页面/API、quant_workflow 脚本、Qlib 接入和 china-astock-quant 可复用能力，输出功能入口和缺口清单。',
      },
      {
        type: 'run_china_astock_comparison',
        label: '生成 china-astock 策略对比',
        description: '读取 china-astock-quant 既有搜索和赢家评估结果，生成策略对比 CSV/Markdown 报告；只读报告，不连接券商或交易接口。',
      },
      {
        type: 'report_china_astock_indicators',
        label: '生成 china-astock 技术指标报告',
        description: '复用 china-astock-quant 的 SMA/EMA/RSI/ATR/Bollinger 指标函数，读取本地 cleaned 行情生成只读技术面观察报告，不连接券商、不自动下单。',
        params: [
          { key: 'symbols', label: '股票代码', kind: 'text', placeholder: '000001,600519；留空按数量上限抽样' },
          { key: 'lookback_days', label: '回看交易日', kind: 'number', min: 20, placeholder: '120' },
          { key: 'limit', label: '股票数量上限', kind: 'number', min: 1, max: 500, placeholder: '50' },
        ],
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
    description: '运行单模型基线或五模型完整候选流程，并记录 MLflow 实验。',
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
        type: 'run_candidate_pipeline',
        label: '运行五模型完整量化流程',
        description: '依次训练 Linear OLS、Linear Ridge、LightGBM、XGBoost 和 CatBoost；只用 2024-2025 选择前两名及五套交易方法，再用 2026 留出期终审。结果写入 MLflow 和候选流程报告，不自动下单。',
        defaultParams: { provider: 'canonical', dry_run: false },
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
  {
    title: '鲁棒策略',
    description: '独立运行历史时点 CSI800、多窗口滚动开发、2026 单次确认和 60 日纸面跟踪，不覆盖日常候选流程。',
    icon: ShieldCheck,
    jobs: [
      {
        type: 'run_robust_strategy',
        label: '运行鲁棒策略开发流程',
        description: '依次检查 PIT CSI800、历史元数据、数据门槛和不可变 provider，再运行 396 个开发期训练任务。数据门槛未通过时固定停止为 blocked_data。',
        params: [
          { key: 'as_of_date', label: '数据截止日', kind: 'date' },
          { key: 'run_id', label: '运行 ID', kind: 'text', placeholder: '留空自动生成' },
          { key: 'resume', label: '恢复同一运行', kind: 'checkbox' },
          { key: 'skip_membership_build', label: '复用已验证历史股票池', kind: 'checkbox' },
        ],
      },
      {
        type: 'run_robust_confirmation',
        label: '运行冻结策略 2026 确认期',
        description: '只对已冻结且开发期通过的配置运行一次 2026 确认；确认结果不参与重新选模型或策略。',
        params: [
          { key: 'run_id', label: '确认运行 ID', kind: 'text', placeholder: '留空自动生成' },
          { key: 'resume', label: '恢复已停止的同一确认运行', kind: 'checkbox' },
        ],
      },
      {
        type: 'run_robust_shadow_day',
        label: '运行鲁棒策略当日纸面跟踪',
        description: '只加载冻结模型并对指定交易日推理，再从 2026 确认期起完整回放持仓；不重新训练、不补造漏跑日信号。',
        params: [
          { key: 'as_of_date', label: '跟踪交易日', kind: 'date' },
        ],
      },
    ],
  },
  {
    title: 'Qlib研究增强',
    description: '把已有 Qlib/MLflow 产物拆成模型解释、回测明细、对比、因子、滚动训练和风险诊断报告。',
    icon: Activity,
    jobs: [
      {
        type: 'report_qlib_model_explainability',
        label: '生成 Qlib 模型解释报告',
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
        label: '生成 Qlib 多模型/多实验对比',
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
        label: '生成 Qlib 风险模型诊断',
        description: '从组合收益、benchmark、换手和最新持仓中估算 beta、集中度、权重上限和风险约束缺口；只读报告。',
        params: [
          { key: 'run_id', label: 'MLflow Run ID', kind: 'text', placeholder: '留空读取最新训练' },
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

const ROBUST_STATUS_LABELS: Record<string, { zh: string; en: string; variant: 'success' | 'danger' | 'warning' | 'info' | 'default' }> = {
  missing: { zh: '未启动', en: 'Not started', variant: 'default' },
  blocked_data: { zh: '数据阻断', en: 'Blocked by data', variant: 'warning' },
  development_failed: { zh: '开发未通过', en: 'Development failed', variant: 'danger' },
  research_candidate_ready: { zh: '研究候选已冻结', en: 'Research candidate frozen', variant: 'info' },
  shadow_observation: { zh: '60日纸面跟踪中', en: '60-day shadow tracking', variant: 'info' },
  robust_strategy_qualified: { zh: '鲁棒策略合格', en: 'Robust strategy qualified', variant: 'success' },
};

function robustStatusLabel(status: string | undefined, language: 'zh' | 'en') {
  const meta = ROBUST_STATUS_LABELS[status || 'missing'] || {
    zh: status || '未知',
    en: status || 'Unknown',
    variant: 'default' as const,
  };
  return { label: meta[language], variant: meta.variant };
}

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
  mlflowUrl: string,
): DailyGuideStep[] {
  const updateJob = latestJobForTypes(latestJobs, [
    'run_daily_update',
    'prepare_baostock_refresh',
    'download_baostock',
    'download_eastmoney',
    'download_akshare',
    'download_jqdata',
  ]);
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
        { kind: 'job', label: '全链路', jobType: 'run_daily_update' },
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
        { kind: 'external', label: 'MLflow', href: mlflowUrl },
      ],
    },
  ];
}

function workflowErrorMessage(err: unknown): string {
  const parsed = getParsedApiError(err);
  return parsed.message || parsed.rawMessage || '请求未成功完成，请稍后重试。';
}

function getBackendJobInfo(quantStatus: QuantStatus | null, jobType: string): {
  catalogLoaded: boolean;
  info?: QuantJobTypeInfo;
  isEnabled: boolean;
  isDisabled: boolean;
  allowedParamKeys: Set<string> | null;
} {
  const info = quantStatus?.availableJobTypes?.[jobType];
  const catalogLoaded = quantStatus !== null;
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
  allowedParamKeys?: Set<string> | null;
}> = ({ job, values, onChange, allowedParamKeys }) => {
  const fields = (job.params || []).filter((field) => isAllowedParam(field.key, allowedParamKeys));
  if (!fields.length) return null;

  return (
    <div className="mt-3 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
      {fields.map((field) => {
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
  const { language, t } = useUiLanguage();
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
      dataSources: quantStatus.dataSources,
      qlibProvider: quantStatus.qlibProvider,
      latestTraining: quantStatus.latestTraining,
      cashControlBacktest: quantStatus.cashControlBacktest,
      qlibPredictions: quantStatus.qlibPredictions,
      qlibResearch: quantStatus.qlibResearch,
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
    const backendJobInfo = getBackendJobInfo(quantStatus, jobType);
    if (!backendJobInfo.catalogLoaded) {
      setError('任务白名单仍在加载，请稍后再试。');
      return;
    }
    if (!backendJobInfo.isEnabled) {
      setError(`后端未启用任务：${workflowJob.label}`);
      return;
    }
    if (submittingJobTypesRef.current.has(jobType)) {
      return;
    }
    submittingJobTypesRef.current.add(jobType);
    setSubmittingTypes((current) => ({ ...current, [jobType]: true }));
    setError(null);
    try {
      const params = explicitParams
        ? filterJobParams(explicitParams, backendJobInfo.allowedParamKeys)
        : buildJobParams(workflowJob, jobParams[jobType], backendJobInfo.allowedParamKeys);
      const job = await quantApi.submitJob(jobType, params);
      setJobs((current) => [job, ...current.filter((item) => item.jobId !== job.jobId)]);
      setQuantStatus((current) => current
        ? {
            ...current,
            activeCount: isActiveJob(job) ? current.activeCount + 1 : current.activeCount,
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
  const robustStrategy = workflowStatus?.robustStrategy;
  const robustMeta = robustStatusLabel(robustStrategy?.status, language);
  const robustProgress = robustStrategy?.progress;
  const robustProgressPercent = robustProgress && robustProgress.expectedDateCount > 0
    ? Math.min(100, Math.round((robustProgress.completedDateCount / robustProgress.expectedDateCount) * 100))
    : null;
  const shadowDaily = robustStrategy?.shadowDaily;
  const shadowDailyFailed = shadowDaily?.status === 'development_failed';
  const dailyGuideSteps = useMemo(
    () => buildDailyGuideSteps(workflowStatus, latestJobs, mlflowUrl),
    [workflowStatus, latestJobs, mlflowUrl],
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
              {trainable
                ? (language === 'zh' ? '当前数据可训练' : 'Data ready for training')
                : loading
                  ? (language === 'zh' ? '正在读取数据状态' : 'Loading data status')
                  : (language === 'zh' ? '数据暂不可训练' : 'Data not ready for training')}
            </p>
            <p className="mt-2 text-sm leading-5 text-secondary-text">
              {workflowStatus?.trainingReadiness?.summary
                || (language === 'zh'
                  ? '运行质量报告后，这里会显示训练准备状态。'
                  : 'Run a data quality report to see training readiness here.')}
            </p>
          </Card>

          <Card variant="bordered" padding="md" className="min-h-[132px]">
            <span className="label-uppercase">Qlib Provider</span>
            <p className="mt-3 text-lg font-semibold text-foreground">
              {workflowStatus?.qlibProvider?.calendarRange || '-'}
            </p>
            <p className="mt-2 text-sm text-secondary-text">
              {language === 'zh' ? '股票池' : 'Universe'} {formatNumber(workflowStatus?.qlibProvider?.instruments)} · {language === 'zh' ? '日历' : 'Calendar'} {formatNumber(workflowStatus?.qlibProvider?.calendarDays)}
            </p>
          </Card>

          <Card variant="bordered" padding="md" className="min-h-[132px]">
            <span className="label-uppercase">Jobs</span>
            <p className="mt-3 text-lg font-semibold text-foreground">
              {language === 'zh'
                ? `${quantStatus?.activeCount ?? 0} 个运行中`
                : `${quantStatus?.activeCount ?? 0} running`}
            </p>
            <p className="mt-2 text-sm text-secondary-text">
              {language === 'zh'
                ? `最近任务 ${jobs.length} 条，状态和日志保存在本地 reports/jobs。`
                : `${jobs.length} recent jobs; status and logs are stored locally under reports/jobs.`}
            </p>
          </Card>

          <Card variant="bordered" padding="md" className="min-h-[132px]">
            <span className="label-uppercase">Latest Training</span>
            <p className="mt-3 truncate text-lg font-semibold text-foreground">
              {latestTraining?.experimentName || (language === 'zh' ? '暂无训练记录' : 'No training runs yet')}
            </p>
            <p className="mt-2 truncate text-sm text-secondary-text">
              Run ID: {latestTraining?.runId || '-'}
            </p>
          </Card>

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
        </section>

        <section className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_380px]">
          <div className="flex flex-col gap-4">
            <Card variant="bordered" padding="md">
              <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                <div>
                  <span className="label-uppercase">Robust CSI800</span>
                  <h2 className="mt-2 text-base font-semibold text-foreground">
                    {language === 'zh' ? '鲁棒策略生命周期' : 'Robust strategy lifecycle'}
                  </h2>
                  <p className="mt-1 text-sm leading-5 text-secondary-text">
                    {language === 'zh'
                      ? '开发期、2026 单次确认和未来 60 个交易日纸面跟踪分开显示；只有最后一项也通过才算合格。'
                      : 'Development, the one-time 2026 confirmation, and 60 future trading days are reported separately.'}
                  </p>
                </div>
                <Badge variant={robustMeta.variant}>{robustMeta.label}</Badge>
              </div>
              <div className="mt-4 grid gap-3 sm:grid-cols-3">
                <div className="border-l-2 border-subtle pl-3">
                  <p className="text-xs text-muted-text">Run ID</p>
                  <p className="mt-1 truncate font-mono text-sm text-foreground">{robustStrategy?.runId || '-'}</p>
                </div>
                <div className="border-l-2 border-subtle pl-3">
                  <p className="text-xs text-muted-text">{language === 'zh' ? '当前/失败阶段' : 'Current/failed stage'}</p>
                  <p className="mt-1 truncate font-mono text-sm text-foreground">
                    {robustStrategy?.currentStage || robustStrategy?.failedStage || '-'}
                  </p>
                </div>
                <div className="border-l-2 border-subtle pl-3">
                  <p className="text-xs text-muted-text">{language === 'zh' ? '原因' : 'Reason'}</p>
                  <p className="mt-1 break-words font-mono text-sm text-foreground">{robustStrategy?.reasonCode || '-'}</p>
                </div>
              </div>
              {robustProgress && robustProgressPercent !== null ? (
                <div data-testid="robust-stage-progress" className="mt-4 border-y border-subtle py-3">
                  <div className="flex items-center justify-between gap-3 text-xs">
                    <span className="text-secondary-text">
                      {language === 'zh' ? '历史股票池构建进度' : 'Historical universe progress'}
                    </span>
                    <span className="font-mono text-foreground">
                      {robustProgress.completedDateCount} / {robustProgress.expectedDateCount}
                    </span>
                  </div>
                  <div
                    role="progressbar"
                    aria-label={language === 'zh' ? '历史股票池构建进度' : 'Historical universe progress'}
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-valuenow={robustProgressPercent}
                    className="mt-2 h-2 overflow-hidden rounded-full bg-subtle"
                  >
                    <div
                      className="h-full bg-cyan transition-[width] duration-300"
                      style={{ width: `${robustProgressPercent}%` }}
                    />
                  </div>
                </div>
              ) : null}
              {(robustStrategy?.stages || []).length > 0 ? (
                <div className="mt-4 divide-y divide-subtle border-y border-subtle">
                  {(robustStrategy?.stages || []).map((stage) => (
                    <div key={stage.name} className="flex flex-wrap items-center justify-between gap-2 py-2 text-xs">
                      <span className="font-mono text-foreground">{stage.name}</span>
                      <span className="text-secondary-text">{stage.status}{stage.reasonCode ? ` · ${stage.reasonCode}` : ''}</span>
                    </div>
                  ))}
                </div>
              ) : null}
              {shadowDaily && shadowDaily.status !== 'missing' ? (
                <div
                  data-testid="robust-shadow-daily"
                  className={`mt-4 border-l-2 px-3 py-2 ${shadowDailyFailed ? 'border-danger bg-danger/5' : 'border-cyan bg-cyan/5'}`}
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <p className="text-xs font-semibold text-foreground">
                      {language === 'zh' ? '最近纸面任务' : 'Latest shadow task'}
                    </p>
                    <Badge variant={shadowDailyFailed ? 'danger' : 'info'} size="sm" className="shadow-none">
                      {shadowDailyFailed
                        ? (language === 'zh' ? '失败' : 'Failed')
                        : (language === 'zh' ? '已记录' : 'Recorded')}
                    </Badge>
                  </div>
                  <p className="mt-1 text-xs text-secondary-text">
                    <span className="font-mono">{shadowDaily.asOf || '-'}</span>
                    {shadowDaily.reasonCode ? (
                      <>
                        <span> · </span>
                        <span className="break-words font-mono">{shadowDaily.reasonCode}</span>
                      </>
                    ) : null}
                  </p>
                </div>
              ) : null}
              <div className="mt-4 flex flex-wrap gap-2">
                <Link to="/backtest" className="btn-secondary h-9 px-3 text-sm">{language === 'zh' ? '查看鲁棒性矩阵' : 'View robustness matrix'}</Link>
                <a href="/experiments/#/experiments" className="btn-secondary h-9 px-3 text-sm">{language === 'zh' ? '查看鲁棒实验' : 'View robust experiments'}</a>
              </div>
            </Card>
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
                      const backendJobInfo = getBackendJobInfo(quantStatus, job.type);
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
                              <JobStatusBadge job={latestJob} />
                              {backendJobInfo.isDisabled ? (
                                <Badge variant="warning" size="sm" className="shadow-none">
                                  后端未启用
                                </Badge>
                              ) : null}
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
                              data-testid={`workflow-run-${job.type}`}
                              className="btn-primary h-9 min-w-[8rem] gap-2 px-3 text-sm"
                              onClick={() => void submitJob(job)}
                              disabled={isSubmitting || !backendJobInfo.catalogLoaded || !backendJobInfo.isEnabled}
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
                              const backendJobInfo = getBackendJobInfo(quantStatus, action.jobType);
                              return (
                                <span key={`${step.id}-${action.jobType}-${action.label}`} className="inline-flex items-center gap-1.5">
                                  <button
                                    type="button"
                                    className="inline-flex h-7 items-center gap-1.5 rounded-lg border border-cyan/25 px-2 text-xs font-medium text-cyan transition-colors hover:bg-cyan/10 disabled:cursor-not-allowed disabled:opacity-60"
                                    onClick={() => void submitGuideJob(action.jobType, action.params)}
                                    disabled={isSubmitting || !backendJobInfo.catalogLoaded || !backendJobInfo.isEnabled}
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
