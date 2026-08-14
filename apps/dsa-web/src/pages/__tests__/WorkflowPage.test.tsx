import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { createParsedApiError } from '../../api/error';
import { UiLanguageProvider } from '../../contexts/UiLanguageContext';
import { UI_LANGUAGE_STORAGE_KEY } from '../../utils/uiLanguage';
import OtherPage from '../OtherPage';
import WorkflowPage, { QuantTaskPage } from '../WorkflowPage';

type FakeEventSourceListener = (event: MessageEvent) => void;

class FakeEventSource {
  static instances: FakeEventSource[] = [];

  readonly url: string;

  readonly init?: EventSourceInit;

  readonly listeners: Record<string, FakeEventSourceListener[]> = {};

  onopen: (() => void) | null = null;

  onerror: (() => void) | null = null;

  closed = false;

  constructor(url: string, init?: EventSourceInit) {
    this.url = url;
    this.init = init;
    FakeEventSource.instances.push(this);
  }

  addEventListener(eventName: string, listener: FakeEventSourceListener) {
    this.listeners[eventName] = [...(this.listeners[eventName] || []), listener];
  }

  removeEventListener(eventName: string, listener: FakeEventSourceListener) {
    this.listeners[eventName] = (this.listeners[eventName] || []).filter((item) => item !== listener);
  }

  close() {
    this.closed = true;
  }

  emit(eventName: string, data: unknown) {
    for (const listener of this.listeners[eventName] || []) {
      listener(new MessageEvent(eventName, { data: JSON.stringify(data) }));
    }
  }

  open() {
    this.onopen?.();
  }

  fail() {
    this.onerror?.();
  }
}

const { getQuantStatus, getQuantJobTypes, getResearchStatus, listQuantJobs, submitQuantJob, getQuantJobLog, getQuantJobArtifactText, cancelQuantJob } = vi.hoisted(() => ({
  getQuantStatus: vi.fn(),
  getQuantJobTypes: vi.fn(),
  getResearchStatus: vi.fn(),
  listQuantJobs: vi.fn(),
  submitQuantJob: vi.fn(),
  getQuantJobLog: vi.fn(),
  getQuantJobArtifactText: vi.fn(),
  cancelQuantJob: vi.fn(),
}));

vi.mock('../../api/quant', () => ({
  QUANT_API_CONTRACT_VERSION: 'quant-workbench-v2',
  quantApi: {
    getStatus: (options?: { includeResearch?: boolean }) => getQuantStatus(options),
    getJobTypes: () => getQuantJobTypes(),
    listJobs: () => listQuantJobs(),
    submitJob: (jobType: string, params?: Record<string, unknown>) => submitQuantJob(jobType, params),
    getJobLog: (jobId: string) => getQuantJobLog(jobId),
    getJobArtifactText: (jobId: string, artifactIndex: number) => getQuantJobArtifactText(jobId, artifactIndex),
    cancelJob: (jobId: string) => cancelQuantJob(jobId),
    getJobArtifactUrl: (jobId: string, artifactIndex: number) => `/api/v1/quant/jobs/${jobId}/artifacts/${artifactIndex}`,
    getJobStreamUrl: () => '/api/v1/quant/jobs/stream',
  },
  parseQuantJobStreamEvent: (data: string) => JSON.parse(data).jobs || [],
}));

vi.mock('../../api/research', () => ({
  researchApi: {
    getStatus: () => getResearchStatus(),
  },
}));

function readyResearchStatus() {
  return {
      generatedAt: '2026-07-03T21:00:00+08:00',
      trainable: true,
      trainingReadiness: { status: 'trainable', summary: 'Qlib provider 已可训练。' },
      trainingPreflight: {
        status: 'passed',
        provider: 'canonical',
        requiredDatasets: ['ashare_daily_qfq', 'cn_index_daily', 'pit_csi800'],
        availableDatasets: [
          'ashare_daily_qfq',
          'cn_index_daily',
          'pit_csi800',
          'qlib_cleaning_plan',
        ],
        summary: 'quant_data 已发布所需数据集，当前 Qlib Provider 轻量预检通过。',
      },
      trainingPreflights: {
        canonical: {
          status: 'passed',
          provider: 'canonical',
          requiredDatasets: ['ashare_daily_qfq', 'cn_index_daily', 'pit_csi800'],
          availableDatasets: [
            'ashare_daily_qfq',
            'cn_index_daily',
            'pit_csi800',
            'qlib_cleaning_plan',
          ],
          summary: 'quant_data 已发布所需数据集，当前 Qlib Provider 轻量预检通过。',
        },
        cleaned: {
          status: 'passed',
          provider: 'cleaned',
          requiredDatasets: [
            'ashare_daily_qfq',
            'cn_index_daily',
            'pit_csi800',
            'qlib_cleaning_plan',
          ],
          availableDatasets: [
            'ashare_daily_qfq',
            'cn_index_daily',
            'pit_csi800',
            'qlib_cleaning_plan',
          ],
          summary: 'quant_data 已发布所需数据集，当前 Qlib Provider 轻量预检通过。',
        },
      },
      dataSources: [],
      qlibProvider: {
        path: '/Users/cq/WorkingProjects/quant_data/qlib/cn_data_baostock_jqdata',
        calendarDays: 1500,
        calendarRange: '2020-01-02 ~ 2026-07-03',
        instruments: 5300,
      },
      providerQuality: {},
      currentProviderIdentities: {
        canonical: {
          provider: 'canonical',
          datasetId: 'ashare_daily_qfq',
          snapshotId: 'snapshot-20260802',
          calendarEnd: '2026-07-24',
          status: 'ready',
        },
        cleaned: {
          provider: 'cleaned',
          datasetId: 'ashare_daily_qfq',
          snapshotId: 'cleaned-snapshot-20260802',
          calendarEnd: '2026-07-24',
          status: 'ready',
        },
      },
      robustSignal: {
        status: 'missing',
        lifecycleStatus: 'not_started',
        dataReadiness: {
          status: 'ready',
          trainable: true,
          summary: 'Robust CSI800 data is ready.',
          summaryZh: '鲁棒 CSI800 数据已验收并绑定当前数据清单。',
          providerKey: 'robust-provider-key',
          dataManifestSha256: 'e'.repeat(64),
        },
      },
      latestTraining: {
        status: 'completed',
        experimentName: 'baseline_lgb_alpha158',
        experimentId: '1',
        runId: 'abc123',
        artifactSha256: {
          'pred.pkl': 'a'.repeat(64),
          'label.pkl': 'b'.repeat(64),
        },
        providerIdentity: {
          providerContract: 'immutable_snapshot_v2',
          datasetId: 'qlib_ashare_v2',
          snapshotId: 'snapshot-20260802',
          manifestSha256: 'c'.repeat(64),
          contentSha256: 'd'.repeat(64),
        },
        metrics: { IC: 0.01 },
      },
      experimentNotebook: {
        status: 'available',
        totalRuns: 1,
        latestNote: 'baseline note',
        runs: [],
      },
    };
}

function renderWorkflowPage(
  surface: 'all' | 'training' | 'other' = 'all',
  embedded = false,
) {
  localStorage.setItem(UI_LANGUAGE_STORAGE_KEY, 'zh');
  return render(
    <UiLanguageProvider>
      <MemoryRouter>
        <QuantTaskPage surface={surface} embedded={embedded} />
      </MemoryRouter>
    </UiLanguageProvider>,
  );
}

describe('WorkflowPage', () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  beforeEach(() => {
    localStorage.clear();
    getQuantStatus.mockReset();
    getQuantJobTypes.mockReset();
    getResearchStatus.mockReset();
    listQuantJobs.mockReset();
    submitQuantJob.mockReset();
    getQuantJobLog.mockReset();
    getQuantJobArtifactText.mockReset();
    cancelQuantJob.mockReset();
    FakeEventSource.instances = [];
    Object.defineProperty(window, 'EventSource', {
      value: FakeEventSource,
      configurable: true,
    });

    getResearchStatus.mockResolvedValue(readyResearchStatus());

    const availableJobTypes = {
        prepare_baostock_refresh: { label: '准备 Baostock 增量清单', allowedParams: ['target_end', 'limit'] },
        download_baostock: { label: '下载 Baostock 长历史', allowedParams: ['start', 'end', 'sleep', 'limit', 'retries', 'timeout', 'max_consecutive_failures', 'max_requests', 'overlap_days', 'retry_failed', 'retry_empty', 'retry_timeout', 'incremental'] },
        download_eastmoney: { label: '下载东方财富日线', allowedParams: ['start', 'end', 'sleep', 'limit', 'retries', 'timeout', 'max_consecutive_failures', 'max_requests', 'retry_failed', 'incremental', 'overlap_days', 'use_system_proxy'] },
        download_akshare: { label: '下载 AkShare 日线', allowedParams: ['start', 'end', 'sleep', 'limit', 'retries', 'timeout', 'max_consecutive_failures', 'max_requests', 'retry_failed', 'incremental', 'overlap_days', 'use_system_proxy'] },
        download_etf_momentum: { label: '更新 ETF 动量数据', allowedParams: ['end'] },
        generate_etf_momentum_report: { label: '生成 ETF 完整策略报告', allowedParams: [] },
        validate_latest_snapshot: { label: '校验新浪最新快照', allowedParams: ['limit', 'timeout', 'use_system_proxy'] },
        download_jqdata: { label: '验证 JQData 连接（dry-run）', allowedParams: ['universe', 'start', 'end', 'limit', 'sleep', 'dry_run'] },
        merge_to_qlib: { label: '生成正式 Qlib Provider', allowedParams: ['no_overwrite', 'smoke'] },
        apply_data_cleaning: { label: '生成正式 cleaned Qlib Provider', allowedParams: ['dry_run'] },
        repair_qlib_provider: { label: '预演旧兼容 Qlib 修复', allowedParams: [] },
        check_qlib_data: { label: '检查 Qlib 数据', allowedParams: [] },
        report_data_anomalies: { label: '生成异常股票和清洗策略', allowedParams: [] },
        report_data_hygiene: { label: '生成数据治理盘点和清理预案', allowedParams: [] },
        report_workbench_capabilities: { label: '生成工作台能力盘点', allowedParams: [] },
        report_market_regime: { label: '生成市场状态和现金仓位信号', allowedParams: ['lookback_days', 'min_symbols', 'limit'] },
        report_cash_control_backtest: { label: '生成现金仓位叠加回测', allowedParams: ['lookback_days', 'min_symbols', 'limit'] },
        report_qlib_predictions: { label: '生成 Qlib 预测摘要', allowedParams: ['run_id', 'topk'] },
        report_qlib_model_explainability: { label: '生成 Qlib 选股模型解释报告', allowedParams: ['run_id', 'topk'] },
        report_qlib_backtest_detail: { label: '生成 Qlib 回测明细报告', allowedParams: ['run_id'] },
        report_qlib_model_comparison: { label: '生成 Qlib 多选股模型/多实验对比', allowedParams: ['max_runs'] },
        report_qlib_factor_analysis: { label: '生成 Qlib 因子分析报告', allowedParams: ['run_id', 'topk'] },
        report_qlib_rolling_training: { label: '生成 Qlib 滚动训练计划', allowedParams: ['config', 'windows', 'train_years', 'valid_months', 'test_months', 'step_months'] },
        report_qlib_risk_model: { label: '生成 Qlib 组合风险诊断', allowedParams: ['run_id'] },
        train_baseline_lgb: { label: '训练 Qlib LightGBM 选股模型基线', allowedParams: ['config', 'experiment_name', 'provider'] },
        train_baseline_lgb_equivalence: { label: '运行真实 LightGBM 网页等价验收', allowedParams: ['provider'] },
        run_candidate_pipeline: { label: '运行五个选股模型候选流程', allowedParams: ['dry_run', 'provider'] },
        prepare_robust_signal_data: { label: '准备冻结的鲁棒 CSI800 数据', allowedParams: [] },
        preview_robust_signal_plan: { label: '预览鲁棒选股信号研究计划', allowedParams: [] },
        run_robust_signal_research: { label: '启动鲁棒选股信号研究', allowedParams: ['run_id', 'smoke'] },
        pause_robust_signal_research: { label: '完成当前子任务后暂停', allowedParams: ['run_id'] },
        resume_robust_signal_research: { label: '恢复鲁棒选股信号研究', allowedParams: ['run_id', 'smoke'] },
        generate_csi300: { label: '预演旧兼容 CSI300 股票池', allowedParams: [] },
        add_index_benchmark: { label: '预演旧兼容 benchmark', allowedParams: [] },
        run_china_astock_comparison: { label: '生成 china-astock 策略对比', allowedParams: [] },
        report_china_astock_indicators: { label: '生成 china-astock 技术指标报告', allowedParams: ['symbols', 'lookback_days', 'limit'] },
        optimize_strategy: { label: '交易策略优化预演', allowedParams: ['dry_run', 'phase', 'topk'] },
      run_dsa_backtest: { label: '运行 DSA 分析准确性回测', allowedParams: ['code', 'eval_window_days', 'force'] },
    };
    const apiContract = {
      version: 'quant-workbench-v2',
      backendBuildId: 'backend-test-123',
      backendStartedAt: '2026-08-02T02:00:00Z',
    };
    getQuantJobTypes.mockResolvedValue({ availableJobTypes, apiContract });

    getQuantStatus.mockResolvedValue({
      apiContract,
      availableJobTypes,
      activeCount: 0,
      latestJobs: {
        train_baseline_lgb: {
          jobId: 'qj_train',
          jobType: 'train_baseline_lgb',
          label: '训练 Qlib LightGBM 选股模型基线',
          status: 'completed',
          params: {},
          command: [],
          cwd: '/Users/cq/WorkingProjects',
          createdAt: '2026-07-03T21:00:00+08:00',
          updatedAt: '2026-07-03T21:10:00+08:00',
          logPath: '/tmp/job.log',
          artifactPaths: ['/Users/cq/WorkingProjects/daily_stock_analysis/quant_workflow/reports/latest_training_result.json'],
        },
      },
      recentJobs: [],
      mlflow: {
        url: 'http://127.0.0.1:6006/',
        status: 'available',
        role: 'experiment_raw_records',
        checkedAt: '2026-07-03T13:10:00Z',
      },
      externalServices: {
        dsa: {
          url: '/',
          status: 'available',
          role: 'local_web_api',
          checkedAt: '2026-07-03T13:10:00Z',
          healthChecked: true,
        },
        mlflow: {
          url: 'http://127.0.0.1:6006/',
          status: 'available',
          role: 'experiment_raw_records',
          checkedAt: '2026-07-03T13:10:00Z',
          healthChecked: true,
        },
        catdesk: {
          url: 'http://127.0.0.1:8765/',
          status: 'available',
          role: 'external_catalyst_dashboard',
          checkedAt: '2026-07-03T13:10:00Z',
          healthChecked: true,
        },
      },
    });
    listQuantJobs.mockResolvedValue({
      total: 1,
      items: [
        {
          jobId: 'qj_train',
          jobType: 'train_baseline_lgb',
          label: '训练 Qlib LightGBM 选股模型基线',
          status: 'completed',
          params: {},
          command: [],
          cwd: '/Users/cq/WorkingProjects',
          createdAt: '2026-07-03T21:00:00+08:00',
          updatedAt: '2026-07-03T21:10:00+08:00',
          logPath: '/tmp/job.log',
          artifactPaths: ['/Users/cq/WorkingProjects/daily_stock_analysis/quant_workflow/reports/latest_training_result.json'],
        },
      ],
    });
    submitQuantJob.mockResolvedValue({
      jobId: 'qj_check',
      jobType: 'check_qlib_data',
      label: '检查 Qlib 数据',
      status: 'running',
      params: {},
      command: [],
      cwd: '/Users/cq/WorkingProjects',
      createdAt: '2026-07-03T22:00:00+08:00',
      updatedAt: '2026-07-03T22:00:00+08:00',
      logPath: '/tmp/job.log',
      artifactPaths: [],
    });
    getQuantJobLog.mockResolvedValue({
      jobId: 'qj_train',
      content: 'training done',
      truncated: false,
      logPath: '/tmp/job.log',
    });
    getQuantJobArtifactText.mockResolvedValue('# Training Result\n\nIC: 0.01');
  });

  it('keeps the public workflow page focused on model training', async () => {
    localStorage.setItem(UI_LANGUAGE_STORAGE_KEY, 'zh');
    render(
      <UiLanguageProvider>
        <MemoryRouter initialEntries={['/workflow']}>
          <WorkflowPage />
        </MemoryRouter>
      </UiLanguageProvider>,
    );

    expect(await screen.findByRole('heading', { name: '选股模型训练' })).toBeInTheDocument();
    expect(screen.getByTestId('workflow-job-train_baseline_lgb')).toBeInTheDocument();
    expect(screen.getByTestId('workflow-job-run_candidate_pipeline')).toBeInTheDocument();
    expect(screen.queryByTestId('workflow-job-run_daily_update')).not.toBeInTheDocument();
    expect(screen.queryByTestId('workflow-job-report_qlib_predictions')).not.toBeInTheDocument();
    expect(screen.getByRole('link', { name: '数据更新与检查' })).toHaveAttribute('href', '/data');
    expect(screen.getByRole('link', { name: '其他量化任务' })).toHaveAttribute('href', '/other');
    expect(screen.queryByText('Latest Training')).not.toBeInTheDocument();
    expect(screen.queryByText('最近训练')).not.toBeInTheDocument();
    expect(screen.queryByRole('link', { name: '实验记录' })).not.toBeInTheDocument();
    expect(screen.queryByText('训练 Qlib LightGBM 基线')).not.toBeInTheDocument();

    const trainingHeading = screen.getByRole('heading', { name: '训练选股模型' });
    const resultsHeading = screen.getByRole('heading', { name: '可回测的选股模型与交易策略' });
    expect(trainingHeading.compareDocumentPosition(resultsHeading) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();

    const backtestLink = screen.getByRole('link', { name: '回测 baseline_lgb_alpha158 每日 TopK 等权基线' });
    expect(backtestLink).toHaveAttribute(
      'href',
      '/backtest?tab=trained&runId=abc123&modelId=latest&strategyId=baseline_daily&snapshotId=snapshot-20260802',
    );
    expect(getQuantStatus).toHaveBeenCalledWith({ includeResearch: false });
  });

  it('keeps the complete training and run-management workflow when embedded in research', async () => {
    const equivalenceJob = {
      jobId: 'qj_equivalence_hidden',
      jobType: 'train_baseline_lgb_equivalence',
      label: '运行真实 LightGBM 网页等价验收',
      status: 'completed',
      params: { provider: 'canonical' },
      createdAt: '2026-08-02T09:00:00+08:00',
      updatedAt: '2026-08-02T09:05:00+08:00',
      artifactPaths: [],
    };
    const smokeJob = {
      jobId: 'qj_smoke_hidden',
      jobType: 'run_robust_signal_research',
      label: '启动鲁棒选股信号研究',
      status: 'completed',
      params: { smoke: true },
      createdAt: '2026-08-02T10:00:00+08:00',
      updatedAt: '2026-08-02T10:05:00+08:00',
      artifactPaths: [],
    };
    listQuantJobs.mockResolvedValueOnce({
      total: 2,
      items: [equivalenceJob, smokeJob],
    });
    getResearchStatus.mockResolvedValueOnce({
      ...readyResearchStatus(),
      robustSignal: {
        status: 'missing',
        lifecycleStatus: 'not_started',
        dataReadiness: {
          status: 'blocked_data',
          trainable: false,
          summaryZh: 'CSI800 数据状态待处理。',
        },
        smokeRun: {
          status: 'signal_diagnosis_complete',
          lifecycleStatus: 'completed',
          runId: 'robust_signal_smoke_hidden',
          currentStage: 'smoke_complete',
          pauseState: 'completed',
          completedCount: 1,
          expectedCount: 1,
          testOnly: true,
        },
      },
    });

    renderWorkflowPage('training', true);

    expect(await screen.findByRole('heading', { name: 'CSI800 选股信号稳定性检验' })).toBeInTheDocument();
    expect(screen.getByTestId('workflow-job-train_baseline_lgb')).toBeInTheDocument();
    expect(screen.getByTestId('workflow-job-run_candidate_pipeline')).toBeInTheDocument();
    expect(screen.getByTestId('workflow-job-train_baseline_lgb_equivalence')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: '运行正式全量数据验收' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: '运行鲁棒信号单季度测试' })).toBeInTheDocument();
    expect(screen.getByText('最近单季度测试')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '恢复鲁棒信号研究' })).toBeInTheDocument();
    expect(screen.getByLabelText('构建信息')).toBeInTheDocument();
    expect(screen.getByText('qj_equivalence_hidden')).toBeInTheDocument();
    expect(screen.getByText('qj_smoke_hidden')).toBeInTheDocument();
    expect(screen.getAllByRole('button', { name: '日志' })).not.toHaveLength(0);
    expect(screen.getByRole('heading', { name: '最近任务' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: '任务日志' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '预览 latest_training_result.json' })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'latest_training_result.json' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '预览鲁棒信号研究计划' })).toBeEnabled();
    expect(screen.getByRole('button', { name: '正式启动鲁棒信号研究' })).toBeDisabled();
  });

  it('disables a job run button while the same job type is active', async () => {
    const runningBaselineJob = {
      jobId: 'qj_train_running',
      jobType: 'train_baseline_lgb',
      label: '训练 Qlib LightGBM 选股模型基线',
      status: 'running',
      params: { provider: 'canonical' },
      command: [],
      cwd: '/Users/cq/WorkingProjects',
      createdAt: '2026-08-02T17:45:00+08:00',
      updatedAt: '2026-08-02T17:45:01+08:00',
      logPath: '/tmp/job.log',
      artifactPaths: [],
    };
    getQuantStatus.mockResolvedValue({
      availableJobTypes: {},
      activeCount: 1,
      latestJobs: { train_baseline_lgb: runningBaselineJob },
      recentJobs: [runningBaselineJob],
    });
    listQuantJobs.mockResolvedValue({ total: 1, items: [runningBaselineJob] });

    renderWorkflowPage('training');

    expect(
      await screen.findByRole('button', { name: '取消 训练 Qlib LightGBM 选股模型基线' }),
    ).toBeEnabled();
    expect(screen.getByTestId('workflow-run-train_baseline_lgb')).toBeDisabled();
  });

  it('blocks formal training when the data gate fails but still allows the explicit dry-run preview', async () => {
    const status = readyResearchStatus();
    getResearchStatus.mockResolvedValue({
      ...status,
      trainable: false,
      trainingReadiness: {
        status: 'blocked_data',
        summary: 'SH000300 benchmark is missing or incomplete (8710 missing days)',
      },
      trainingPreflight: {
        ...status.trainingPreflight,
        status: 'blocked',
        summary: 'SH000300 benchmark is missing or incomplete (8710 missing days)',
      },
      trainingPreflights: Object.fromEntries(
        Object.entries(status.trainingPreflights).map(([provider, preflight]) => [
          provider,
          {
            ...preflight,
            status: 'blocked',
            summary: 'SH000300 benchmark is missing or incomplete (8710 missing days)',
          },
        ]),
      ),
      providerQuality: {
        ...status.providerQuality,
        canonical: {
          provider: 'canonical',
          datasetId: 'qlib_ashare_v2',
          snapshotId: 'canonical-snapshot-blocked',
          trainable: false,
          trainingReadiness: {
            status: 'blocked_data',
            summary: 'SH000300 benchmark is missing or incomplete (8710 missing days)',
          },
        },
        cleaned: {
          trainable: false,
          structuralStatus: 'blocked',
          trainingReadiness: {
            status: 'blocked_data',
            summary: 'SH000300 benchmark is missing or incomplete (8710 missing days)',
          },
        },
      },
    });
    localStorage.setItem(UI_LANGUAGE_STORAGE_KEY, 'zh');
    render(
      <UiLanguageProvider>
        <MemoryRouter initialEntries={['/workflow']}>
          <WorkflowPage />
        </MemoryRouter>
      </UiLanguageProvider>,
    );

    expect(await screen.findByText('数据暂不可训练')).toBeInTheDocument();
    expect(screen.getByText('正式训练已锁定')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: '去数据页修复' })).toHaveAttribute('href', '/data');

    const baselineRun = screen.getByTestId('workflow-run-train_baseline_lgb');
    const candidateRun = screen.getByTestId('workflow-run-run_candidate_pipeline');
    expect(baselineRun).toBeDisabled();
    expect(candidateRun).toBeDisabled();

    fireEvent.click(screen.getByLabelText('运行五个选股模型候选流程 只做小样本预演'));
    expect(candidateRun).toBeEnabled();
    fireEvent.click(candidateRun);

    await waitFor(() => {
      expect(submitQuantJob).toHaveBeenCalledWith('run_candidate_pipeline', {
        provider: 'canonical',
        dry_run: true,
      });
    });
    expect(submitQuantJob).not.toHaveBeenCalledWith('train_baseline_lgb', expect.anything());
  });

  it('launches the isolated real LightGBM equivalence run with only a provider choice', async () => {
    renderWorkflowPage('training');

    const card = await screen.findByTestId(
      'workflow-job-train_baseline_lgb_equivalence',
    );
    expect(
      within(card).getByText(/真实训练.*测试结果不会进入正式候选、最新结果或普通回测列表/),
    ).toBeInTheDocument();
    expect(within(card).getByText(/服务器固定短周期配置/)).toBeInTheDocument();
    expect(within(card).queryByLabelText(/配置文件/)).not.toBeInTheDocument();
    expect(within(card).queryByLabelText(/实验名称/)).not.toBeInTheDocument();

    fireEvent.click(
      screen.getByTestId('workflow-run-train_baseline_lgb_equivalence'),
    );

    await waitFor(() => {
      expect(submitQuantJob).toHaveBeenCalledWith(
        'train_baseline_lgb_equivalence',
        { provider: 'canonical' },
      );
    });
  });

  it('does not expose a completed historical quality report on the research page', async () => {
    const completedQualityJob = {
      jobId: 'qj_quality_complete',
      jobType: 'report_data_quality',
      label: '生成数据质量报告',
      status: 'completed',
      params: {},
      createdAt: '2026-08-02T09:00:00+08:00',
      completedAt: '2026-08-02T09:05:00+08:00',
      updatedAt: '2026-08-02T09:05:00+08:00',
      artifactPaths: ['data_quality_report_latest.json'],
    };
    const status = readyResearchStatus();
    getResearchStatus.mockResolvedValueOnce({
      ...status,
      trainable: true,
      trainingReadiness: status.trainingReadiness,
      providerQuality: {},
    });
    getQuantStatus.mockResolvedValue({
      availableJobTypes: {},
      activeCount: 0,
      latestJobs: { report_data_quality: completedQualityJob },
      recentJobs: [completedQualityJob],
    });
    listQuantJobs.mockResolvedValueOnce({ total: 1, items: [completedQualityJob] });

    renderWorkflowPage('training');

    expect(await screen.findByText('实时检查通过')).toBeInTheDocument();
    expect(screen.queryByText(/数据验收/)).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /完整.*验收/ })).not.toBeInTheDocument();
    expect(screen.getByTestId('workflow-run-train_baseline_lgb')).toBeEnabled();
    expect(submitQuantJob).not.toHaveBeenCalledWith('report_data_quality', {});
  });

  it('does not report training preflight as blocked while status is still loading', () => {
    getResearchStatus.mockReturnValueOnce(new Promise(() => undefined));
    getQuantStatus.mockReturnValueOnce(new Promise(() => undefined));

    renderWorkflowPage('training');

    expect(screen.getByText('正在读取数据状态')).toBeInTheDocument();
    expect(screen.getByText('实时检查：读取中')).toBeInTheDocument();
    expect(screen.queryByText('实时检查：未通过')).not.toBeInTheDocument();
  });

  it('shows structured model-training progress and runner health', async () => {
    const candidateJob = {
      jobId: 'qj_candidate_progress',
      jobType: 'run_candidate_pipeline',
      label: '运行五个选股模型候选流程',
      status: 'running',
      params: { provider: 'canonical' },
      pid: 43210,
      runnerAlive: true,
      progress: {
        stage: 'models',
        completed: 2,
        total: 5,
        percent: 40,
        current: 'lightgbm',
        currentArtifact: 'pred.pkl',
        elapsedSeconds: 120,
        etaSeconds: 180,
      },
      createdAt: '2026-08-02T10:00:00+08:00',
      startedAt: '2026-08-02T10:00:00+08:00',
      updatedAt: '2026-08-02T10:02:00+08:00',
      artifactPaths: [],
    };
    getQuantStatus.mockResolvedValue({
      availableJobTypes: {},
      activeCount: 1,
      latestJobs: { run_candidate_pipeline: candidateJob },
      recentJobs: [candidateJob],
    });
    listQuantJobs.mockResolvedValueOnce({ total: 1, items: [candidateJob] });

    renderWorkflowPage('training');

    expect(await screen.findByText('2 / 5 · 40%')).toBeInTheDocument();
    expect(screen.getByText('阶段：models · 当前：lightgbm')).toBeInTheDocument();
    expect(screen.getByText('已用 2分钟 · 预计剩余 3分钟 · 当前产物 pred.pkl')).toBeInTheDocument();
    expect(screen.getByText('Runner PID 43210 · 存活')).toBeInTheDocument();
  });

  it('provides controlled robust-signal plan, smoke, formal, pause and resume actions', async () => {
    getResearchStatus.mockResolvedValueOnce({
      ...readyResearchStatus(),
      robustSignal: {
        status: 'signal_diagnosis_complete',
        lifecycleStatus: 'paused',
        runId: 'robust_signal_20260802_100000',
        currentStage: 'validation',
        pauseState: 'paused',
        nextGate: 'resume_validation',
        completedCount: 17,
        expectedCount: 24,
        configHash: 'config-hash',
        dataManifestHash: 'data-hash',
        gitCommit: 'abcdef123456',
        dataReadiness: {
          status: 'ready',
          trainable: true,
          summaryZh: '鲁棒 CSI800 数据已验收并绑定当前数据清单。',
          providerKey: 'robust-provider-key',
          dataManifestSha256: 'e'.repeat(64),
        },
      },
    });

    renderWorkflowPage('training');

    expect(await screen.findByRole('heading', { name: 'CSI800 选股信号稳定性检验' })).toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: '鲁棒选股信号研究' })).not.toBeInTheDocument();
    expect(screen.getByText('17 / 24 · validation · 已暂停')).toBeInTheDocument();
    expect(screen.getByText('检查点：robust_signal_20260802_100000')).toBeInTheDocument();
    expect(screen.getByText('配置 config-hash · 数据 data-hash · 代码 abcdef123456')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: '预览鲁棒信号研究计划' }));
    await waitFor(() => {
      expect(submitQuantJob).toHaveBeenCalledWith('preview_robust_signal_plan', {});
    });

    fireEvent.click(screen.getByRole('button', { name: '运行鲁棒信号单季度测试' }));
    await waitFor(() => {
      expect(submitQuantJob).toHaveBeenCalledWith('run_robust_signal_research', { smoke: true });
    });

    fireEvent.click(screen.getByRole('button', { name: '恢复鲁棒信号研究' }));
    await waitFor(() => {
      expect(submitQuantJob).toHaveBeenCalledWith('resume_robust_signal_research', {
        run_id: 'robust_signal_20260802_100000',
      });
    });
    expect(screen.getByRole('button', { name: '正式启动鲁棒信号研究' })).toBeEnabled();
    expect(screen.getByRole('button', { name: '完成当前子任务后暂停' })).toBeDisabled();
  });

  it('disables robust-signal launch and resume actions missing from the backend catalog', async () => {
    getResearchStatus.mockResolvedValueOnce({
      ...readyResearchStatus(),
      robustSignal: {
        ...readyResearchStatus().robustSignal,
        status: 'signal_diagnosis_complete',
        lifecycleStatus: 'paused',
        pauseState: 'paused',
        runId: 'robust_signal_20260802_100000',
      },
    });
    getQuantJobTypes.mockResolvedValueOnce({
      availableJobTypes: {
        train_baseline_lgb: { label: '训练 Qlib LightGBM 选股模型基线', allowedParams: ['provider'] },
        train_baseline_lgb_equivalence: { label: '运行真实 LightGBM 网页等价验收', allowedParams: ['provider'] },
        run_candidate_pipeline: { label: '运行五个选股模型候选流程', allowedParams: ['dry_run', 'provider'] },
      },
      apiContract: {
        version: 'quant-workbench-v2',
        backendBuildId: 'backend-test-123',
        backendStartedAt: '2026-08-02T02:00:00Z',
      },
    });

    renderWorkflowPage('training');

    expect(await screen.findByText('后端未启用鲁棒信号研究任务')).toBeInTheDocument();
    expect(screen.getByText(/不可用操作：预览计划、启动研究、协作暂停、恢复研究/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '预览鲁棒信号研究计划' })).toBeDisabled();
    expect(screen.getByRole('button', { name: '运行鲁棒信号单季度测试' })).toBeDisabled();
    expect(screen.getByRole('button', { name: '正式启动鲁棒信号研究' })).toBeDisabled();
    expect(screen.getByRole('button', { name: '恢复鲁棒信号研究' })).toBeDisabled();
  });

  it('disables robust-signal pause when that backend action is not advertised', async () => {
    getResearchStatus.mockResolvedValueOnce({
      ...readyResearchStatus(),
      robustSignal: {
        ...readyResearchStatus().robustSignal,
        status: 'running',
        lifecycleStatus: 'running',
        pauseState: 'running',
        runId: 'robust_signal_20260802_110000',
      },
    });
    getQuantJobTypes.mockResolvedValueOnce({
      availableJobTypes: {
        preview_robust_signal_plan: { label: '预览鲁棒选股信号研究计划', allowedParams: [] },
        run_robust_signal_research: { label: '启动鲁棒选股信号研究', allowedParams: ['run_id', 'smoke'] },
        resume_robust_signal_research: { label: '恢复鲁棒选股信号研究', allowedParams: ['run_id', 'smoke'] },
        train_baseline_lgb: { label: '训练 Qlib LightGBM 选股模型基线', allowedParams: ['provider'] },
        train_baseline_lgb_equivalence: { label: '运行真实 LightGBM 网页等价验收', allowedParams: ['provider'] },
        run_candidate_pipeline: { label: '运行五个选股模型候选流程', allowedParams: ['dry_run', 'provider'] },
      },
      apiContract: {
        version: 'quant-workbench-v2',
        backendBuildId: 'backend-test-123',
        backendStartedAt: '2026-08-02T02:00:00Z',
      },
    });

    renderWorkflowPage('training');

    expect(await screen.findByText('后端未启用鲁棒信号研究任务')).toBeInTheDocument();
    expect(screen.getByText(/不可用操作：协作暂停/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '完成当前子任务后暂停' })).toBeDisabled();
  });

  it('blocks every robust-signal run mode on its own CSI800 readiness contract', async () => {
    getResearchStatus.mockResolvedValueOnce({
      ...readyResearchStatus(),
      robustSignal: {
        status: 'missing',
        lifecycleStatus: 'not_started',
        dataReadiness: {
          status: 'blocked_data',
          trainable: false,
          reasonCode: 'data_manifest_changed',
          summary: 'The robust provider was built from an older data manifest.',
          summaryZh: '鲁棒 CSI800 provider 绑定的是旧数据清单，请重新验收并准备。',
        },
      },
    });

    renderWorkflowPage('training');

    expect(await screen.findByText('鲁棒 CSI800 provider 绑定的是旧数据清单，请重新验收并准备。')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '预览鲁棒信号研究计划' })).toBeEnabled();
    expect(screen.getByRole('button', { name: '运行鲁棒信号单季度测试' })).toBeDisabled();
    expect(screen.getByRole('button', { name: '正式启动鲁棒信号研究' })).toBeDisabled();
    expect(screen.getByRole('link', { name: '去数据页准备鲁棒 CSI800 数据' })).toHaveAttribute('href', '/data');
  });

  it('shows the latest one-quarter test separately from formal robust research', async () => {
    getResearchStatus.mockResolvedValueOnce({
      ...readyResearchStatus(),
      robustSignal: {
        status: 'missing',
        lifecycleStatus: 'not_started',
        dataReadiness: {
          status: 'ready',
          trainable: true,
          summaryZh: '鲁棒 CSI800 数据已验收并绑定当前数据清单。',
        },
        smokeRun: {
          status: 'signal_diagnosis_complete',
          lifecycleStatus: 'completed',
          runId: 'robust_signal_20260802_165707',
          currentStage: 'smoke_complete',
          pauseState: 'completed',
          completedCount: 1,
          expectedCount: 1,
          testOnly: true,
          progress: {
            currentStage: 'smoke_complete',
            completedCount: 1,
            expectedCount: 1,
          },
        },
      },
    });

    renderWorkflowPage('training');

    expect(await screen.findByText('最近单季度测试')).toBeInTheDocument();
    expect(screen.getByText('测试数据，不可用于研究或交易')).toBeInTheDocument();
    expect(screen.getByText('1 / 1 · smoke_complete · 已完成')).toBeInTheDocument();
    expect(screen.getByText('测试检查点：robust_signal_20260802_165707')).toBeInTheDocument();
    expect(screen.getByText('0 / 0 · - · not_started')).toBeInTheDocument();
  });

  it('polls a durable robust smoke runner until its visible status is complete', async () => {
    const running = {
      ...readyResearchStatus(),
      robustSignal: {
        ...readyResearchStatus().robustSignal,
        smokeRun: {
          status: 'signal_diagnosis_complete',
          lifecycleStatus: 'running',
          runId: 'robust_signal_20260802_170000',
          currentStage: 'screening',
          pauseState: 'running',
          completedCount: 0,
          expectedCount: 1,
          testOnly: true,
        },
      },
    };
    const completed = {
      ...running,
      robustSignal: {
        ...running.robustSignal,
        smokeRun: {
          ...running.robustSignal.smokeRun,
          lifecycleStatus: 'completed',
          currentStage: 'smoke_complete',
          pauseState: 'completed',
          completedCount: 1,
        },
      },
    };
    getResearchStatus.mockReset();
    getResearchStatus.mockResolvedValueOnce(running).mockResolvedValue(completed);

    renderWorkflowPage('training');

    const panel = await screen.findByLabelText('最近鲁棒信号单季度测试');
    expect(within(panel).getByText('0 / 1 · screening · 运行中')).toBeInTheDocument();
    await waitFor(() => {
      expect(within(panel).getByText('1 / 1 · smoke_complete · 已完成')).toBeInTheDocument();
    }, { timeout: 5000 });
    expect(getResearchStatus.mock.calls.length).toBeGreaterThanOrEqual(2);
  });

  it('does not let an older research response roll back a newer robust status', async () => {
    renderWorkflowPage('training');
    await screen.findByText('实时检查通过');

    let resolveOlder: (value: unknown) => void = () => undefined;
    let resolveNewer: (value: unknown) => void = () => undefined;
    getResearchStatus
      .mockReturnValueOnce(new Promise((resolve) => {
        resolveOlder = resolve;
      }))
      .mockReturnValueOnce(new Promise((resolve) => {
        resolveNewer = resolve;
      }));

    const refreshButton = screen.getByRole('button', { name: '刷新' });
    fireEvent.click(refreshButton);
    fireEvent.click(refreshButton);
    await waitFor(() => {
      expect(getResearchStatus).toHaveBeenCalledTimes(3);
    });

    const newer = {
      ...readyResearchStatus(),
      robustSignal: {
        ...readyResearchStatus().robustSignal,
        status: 'signal_candidate_ready',
        lifecycleStatus: 'completed',
        runId: 'robust_signal_newer',
        currentStage: 'validation_complete',
        pauseState: 'completed',
        completedCount: 24,
        expectedCount: 24,
      },
    };
    await act(async () => {
      resolveNewer(newer);
    });
    expect(await screen.findByText('24 / 24 · validation_complete · completed')).toBeInTheDocument();

    const older = {
      ...readyResearchStatus(),
      robustSignal: {
        ...readyResearchStatus().robustSignal,
        status: 'signal_diagnosis_complete',
        lifecycleStatus: 'running',
        runId: 'robust_signal_older',
        currentStage: 'screening',
        pauseState: 'running',
        completedCount: 3,
        expectedCount: 24,
      },
    };
    await act(async () => {
      resolveOlder(older);
    });

    expect(screen.getByText('24 / 24 · validation_complete · completed')).toBeInTheDocument();
    expect(screen.queryByText('3 / 24 · screening · 运行中')).not.toBeInTheDocument();
  });

  it('clears the previous research status when the newest refresh fails', async () => {
    renderWorkflowPage('training');
    await screen.findByText('实时检查通过');

    getResearchStatus.mockRejectedValueOnce(new Error('research status unavailable'));
    fireEvent.click(screen.getByRole('button', { name: '刷新' }));

    await waitFor(() => {
      expect(screen.queryByText('实时检查通过')).not.toBeInTheDocument();
    });
  });

  it('lets a human cancel a durable robust run after its launcher job completed', async () => {
    const completedLauncher = {
      jobId: 'qj_robust_launcher',
      jobType: 'run_robust_signal_research',
      label: '启动鲁棒选股信号研究',
      status: 'completed',
      params: { run_id: 'robust_signal_20260802_170000', smoke: false },
      returnCode: 0,
      createdAt: '2026-08-02T17:00:00+08:00',
      startedAt: '2026-08-02T17:00:01+08:00',
      completedAt: '2026-08-02T17:00:11+08:00',
      updatedAt: '2026-08-02T17:00:11+08:00',
      artifactPaths: [],
      canCancel: true,
      progress: {
        runId: 'robust_signal_20260802_170000',
        lifecycleStatus: 'running',
        pauseState: 'running',
      },
    };
    getResearchStatus.mockResolvedValue({
      ...readyResearchStatus(),
      robustSignal: {
        ...readyResearchStatus().robustSignal,
        runId: 'robust_signal_20260802_170000',
        lifecycleStatus: 'running',
        pauseState: 'running',
        currentStage: 'screening',
        completedCount: 1,
        expectedCount: 8,
      },
    });
    getQuantStatus.mockResolvedValueOnce({
      availableJobTypes: {},
      activeCount: 0,
      latestJobs: { run_robust_signal_research: completedLauncher },
      recentJobs: [completedLauncher],
    });
    listQuantJobs.mockResolvedValueOnce({ total: 1, items: [completedLauncher] });
    cancelQuantJob.mockResolvedValueOnce({
      ...completedLauncher,
      status: 'cancelled',
      canCancel: false,
    });

    renderWorkflowPage('training');

    const cancel = await screen.findByRole('button', {
      name: '取消正在运行的鲁棒信号研究',
    });
    expect(cancel).toBeEnabled();
    fireEvent.click(cancel);
    expect(cancelQuantJob).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole('button', { name: '确认取消' }));

    await waitFor(() => {
      expect(cancelQuantJob).toHaveBeenCalledWith('qj_robust_launcher');
    });
  });

  it('shows durable cancellation failure instead of pretending the run stopped', async () => {
    const completedLauncher = {
      jobId: 'qj_robust_cancel_failure',
      jobType: 'run_robust_signal_research',
      label: '启动鲁棒选股信号研究',
      status: 'completed',
      params: { run_id: 'robust_signal_20260802_170500', smoke: false },
      returnCode: 0,
      createdAt: '2026-08-02T17:05:00+08:00',
      completedAt: '2026-08-02T17:05:11+08:00',
      updatedAt: '2026-08-02T17:05:11+08:00',
      artifactPaths: [],
      canCancel: true,
    };
    getResearchStatus.mockResolvedValue({
      ...readyResearchStatus(),
      robustSignal: {
        ...readyResearchStatus().robustSignal,
        runId: 'robust_signal_20260802_170500',
        lifecycleStatus: 'running',
        pauseState: 'running',
      },
    });
    getQuantStatus.mockResolvedValueOnce({
      availableJobTypes: {},
      activeCount: 0,
      latestJobs: { run_robust_signal_research: completedLauncher },
      recentJobs: [completedLauncher],
    });
    listQuantJobs.mockResolvedValueOnce({ total: 1, items: [completedLauncher] });
    cancelQuantJob.mockResolvedValueOnce({
      ...completedLauncher,
      error: '取消未确认：后台训练仍可能运行。',
    });

    renderWorkflowPage('training');

    fireEvent.click(await screen.findByRole('button', {
      name: '取消正在运行的鲁棒信号研究',
    }));
    fireEvent.click(screen.getByRole('button', { name: '确认取消' }));

    expect(await screen.findByText('取消未确认：后台训练仍可能运行。')).toBeInTheDocument();
    expect(screen.getByRole('button', {
      name: '取消正在运行的鲁棒信号研究',
    })).toBeEnabled();
  });

  it('treats a completed launcher with a cancellable runner as active and blocks duplicate launch', async () => {
    const completedLauncher = {
      jobId: 'qj_robust_detached_runner',
      jobType: 'run_robust_signal_research',
      label: '启动鲁棒选股信号研究',
      status: 'completed',
      params: { run_id: 'robust_signal_20260802_180000', smoke: false },
      returnCode: 0,
      createdAt: '2026-08-02T18:00:00+08:00',
      startedAt: '2026-08-02T18:00:01+08:00',
      completedAt: '2026-08-02T18:00:11+08:00',
      updatedAt: '2026-08-02T18:00:11+08:00',
      artifactPaths: [],
      canCancel: true,
      runnerAlive: true,
    };
    getQuantStatus.mockResolvedValue({
      availableJobTypes: {},
      activeCount: 0,
      latestJobs: { run_robust_signal_research: completedLauncher },
      recentJobs: [completedLauncher],
    });
    listQuantJobs.mockResolvedValue({ total: 1, items: [completedLauncher] });

    const view = renderWorkflowPage('all');

    expect(await screen.findByText('1 个运行中')).toBeInTheDocument();
    expect(screen.getByText('qj_robust_detached_runner')).toBeInTheDocument();

    view.rerender(
      <UiLanguageProvider>
        <MemoryRouter>
          <QuantTaskPage surface="training" />
        </MemoryRouter>
      </UiLanguageProvider>,
    );

    expect(await screen.findByRole('button', { name: '运行鲁棒信号单季度测试' })).toBeDisabled();
    expect(screen.getByRole('button', { name: '正式启动鲁棒信号研究' })).toBeDisabled();
  });

  it('pauses and resumes a smoke checkpoint without switching to formal mode', async () => {
    getResearchStatus.mockResolvedValueOnce({
      ...readyResearchStatus(),
      robustSignal: {
        ...readyResearchStatus().robustSignal,
        smokeRun: {
          status: 'signal_diagnosis_complete',
          lifecycleStatus: 'stopped',
          runId: 'robust_signal_20260802_171500',
          currentStage: 'screening',
          pauseState: 'paused',
          completedCount: 1,
          expectedCount: 1,
          testOnly: true,
        },
      },
    });

    renderWorkflowPage('training');

    const resume = await screen.findByRole('button', { name: '恢复鲁棒信号研究' });
    expect(resume).toBeEnabled();
    fireEvent.click(resume);
    await waitFor(() => {
      expect(submitQuantJob).toHaveBeenCalledWith('resume_robust_signal_research', {
        run_id: 'robust_signal_20260802_171500',
        smoke: true,
      });
    });
  });

  it('offers robust CSI800 data preparation on the data page', async () => {
    renderWorkflowPage('all');

    fireEvent.click(await screen.findByRole('button', { name: '运行 准备冻结的鲁棒 CSI800 数据' }));

    await waitFor(() => {
      expect(submitQuantJob).toHaveBeenCalledWith('prepare_robust_signal_data', {});
    });
  });

  it('does not present synthetic or incomplete results as backtest-ready', async () => {
    getResearchStatus.mockResolvedValue({
      ...readyResearchStatus(),
      latestTraining: {
        status: 'completed',
        experimentName: 'synthetic_preview',
      },
      candidatePipeline: {
        status: 'completed',
        dryRun: true,
        experimentName: 'candidate_models_preview',
        models: [
          { id: 'lightgbm', label: 'LightGBM', status: 'completed', runId: null },
        ],
        strategyEvaluations: [
          {
            modelId: 'lightgbm',
            strategyId: 'baseline_daily',
            strategyLabel: '每日 TopK 等权基线',
            strategyConfig: { rebalanceFreq: 1 },
          },
        ],
      },
    });

    renderWorkflowPage('training');

    expect(await screen.findByText('测试数据，不可用于研究或交易')).toBeInTheDocument();
    expect(screen.queryByRole('link', { name: /^回测 / })).not.toBeInTheDocument();
    expect(screen.getAllByText('查看测试结果').length).toBeGreaterThan(0);
    expect(screen.getAllByText(/缺少真实 Run ID/).length).toBeGreaterThan(0);
  });

  it('gates each training card with the selected provider snapshot', async () => {
    const status = readyResearchStatus();
    getResearchStatus.mockResolvedValue({
      ...status,
      trainingPreflights: {
        ...status.trainingPreflights,
        cleaned: {
          ...status.trainingPreflights.cleaned,
          status: 'blocked',
          summary: 'cleaned provider 缺少完整 benchmark',
        },
      },
      currentProviderIdentities: {
        ...status.currentProviderIdentities,
        cleaned: {
          ...status.currentProviderIdentities.cleaned,
          snapshotId: 'cleaned-snapshot-blocked',
        },
      },
      providerQuality: {
        ...status.providerQuality,
        cleaned: {
          snapshotId: 'cleaned-snapshot-blocked',
          trainable: false,
          structuralStatus: 'blocked',
          trainingReadiness: { status: 'blocked', summary: 'cleaned provider 缺少完整 benchmark' },
        },
      },
    });
    renderWorkflowPage('training');

    const baselineCard = await screen.findByTestId('workflow-job-train_baseline_lgb');
    expect(
      within(baselineCard).getByLabelText('训练 Qlib LightGBM 选股模型基线 数据版本'),
    ).toHaveValue('canonical');
    expect(within(baselineCard).getByText(/snapshot-20260802/)).toBeInTheDocument();
    expect(screen.getByTestId('workflow-run-train_baseline_lgb')).toBeEnabled();

    fireEvent.change(
      within(baselineCard).getByLabelText('训练 Qlib LightGBM 选股模型基线 数据版本'),
      { target: { value: 'cleaned' } },
    );

    expect(within(baselineCard).getByText(/cleaned-snapshot-blocked/)).toBeInTheDocument();
    expect(within(baselineCard).getByText(/cleaned provider 缺少完整 benchmark/)).toBeInTheDocument();
    expect(screen.getByTestId('workflow-run-train_baseline_lgb')).toBeDisabled();
  });

  it('keeps remaining quant tasks on the legacy other route', async () => {
    localStorage.setItem(UI_LANGUAGE_STORAGE_KEY, 'zh');
    render(
      <UiLanguageProvider>
        <MemoryRouter initialEntries={['/other']}>
          <OtherPage />
        </MemoryRouter>
      </UiLanguageProvider>,
    );

    expect(await screen.findByRole('heading', { name: '其他量化任务' })).toBeInTheDocument();
    expect(screen.getByTestId('workflow-job-report_qlib_predictions')).toBeInTheDocument();
    expect(screen.getByTestId('workflow-job-report_workbench_capabilities')).toBeInTheDocument();
    expect(screen.getByTestId('workflow-job-generate_etf_momentum_report')).toBeInTheDocument();
    expect(screen.queryByTestId('workflow-job-run_daily_update')).not.toBeInTheDocument();
    expect(screen.queryByTestId('workflow-job-train_baseline_lgb')).not.toBeInTheDocument();
  });

  it('renders a Chinese web-first quant workflow console', async () => {
    renderWorkflowPage();

    expect(screen.getByRole('heading', { name: '量化任务中心' })).toBeInTheDocument();
    expect(await screen.findByText('实时检查通过')).toBeInTheDocument();
    expect(screen.getByText('外部服务状态')).toBeInTheDocument();
    expect(screen.getByText('本地研究工作台')).toBeInTheDocument();
    expect(screen.getByText('MLflow 原始记录')).toBeInTheDocument();
    expect(screen.getByText('CatDesk')).toBeInTheDocument();
    expect(screen.getByText('美股催化观察台')).toBeInTheDocument();
    expect(screen.getAllByText('可用').length).toBeGreaterThan(0);
    expect(screen.getByRole('link', { name: '打开 MLflow，新窗口' })).toHaveAttribute('href', '/experiments/');
    expect(screen.getByRole('link', { name: '打开 CatDesk，新窗口' })).toHaveAttribute('href', 'http://127.0.0.1:8765/');
    expect(screen.getByRole('link', { name: 'MLflow' })).toHaveAttribute('href', '/experiments/');
    expect(screen.queryByText('/Users/cq/WorkingProjects')).not.toBeInTheDocument();
    expect(screen.queryByText('/tmp/job.log')).not.toBeInTheDocument();
    expect(screen.getByText('2020-01-02 ~ 2026-07-03')).toBeInTheDocument();
    expect(screen.getAllByText('更新数据')).toHaveLength(2);
    expect(screen.getByText('下载东方财富日线')).toBeInTheDocument();
    expect(screen.getByText('匿名抓取东财 qfq 日线，写入 eastmoney_qfq 作为 Baostock/JQData 的冗余源。')).toBeInTheDocument();
    expect(screen.getByText('下载 AkShare 日线')).toBeInTheDocument();
    expect(screen.getByText('调用 AkShare stock_zh_a_hist(qfq)，写入 akshare_qfq 作为第二条免费冗余源。')).toBeInTheDocument();
    expect(screen.getAllByText('Qlib Provider').length).toBeGreaterThan(0);
    expect(screen.getByText('校验新浪最新快照')).toBeInTheDocument();
    expect(screen.getByText('抽样读取新浪行情快照，对比合并后的最新收盘价；只生成校验报告，不参与训练。')).toBeInTheDocument();
    expect(screen.getByText('预演旧兼容 CSI300 股票池')).toBeInTheDocument();
    expect(screen.getByText('预演旧兼容 benchmark')).toBeInTheDocument();
    expect(screen.queryByLabelText('修复 Qlib Provider 只检查不写入')).not.toBeInTheDocument();
    expect(screen.getByText('生成正式 cleaned Qlib Provider')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /生成数据质量报告/ })).not.toBeInTheDocument();
    expect(screen.getByText('选股模型训练与交易策略验证')).toBeInTheDocument();
    expect(screen.getByText('运行五个选股模型候选流程')).toBeInTheDocument();
    expect(screen.getByTestId('workflow-job-run_candidate_pipeline')).toBeInTheDocument();
    expect(screen.getByTestId('workflow-run-run_candidate_pipeline')).toHaveAccessibleName('运行 运行五个选股模型候选流程');
    expect(screen.getByLabelText('运行五个选股模型候选流程 数据版本')).toBeInTheDocument();
    expect(screen.getAllByRole('option', { name: 'Qlib Provider（canonical）' })).toHaveLength(3);
    expect(screen.getAllByRole('option', { name: 'Qlib Provider（cleaned）' })).toHaveLength(3);
    expect(screen.getByLabelText('运行五个选股模型候选流程 只做小样本预演')).toBeInTheDocument();
    expect(screen.getByText('生成市场状态和现金仓位信号')).toBeInTheDocument();
    expect(screen.getByLabelText('生成市场状态和现金仓位信号 回看交易日')).toBeInTheDocument();
    expect(screen.getByText('生成现金仓位叠加回测')).toBeInTheDocument();
    expect(screen.getByLabelText('生成现金仓位叠加回测 回看交易日')).toBeInTheDocument();
    expect(screen.getByText('生成 Qlib 预测摘要')).toBeInTheDocument();
    expect(screen.getByLabelText('生成 Qlib 预测摘要 TopK')).toBeInTheDocument();
    expect(screen.getByText('Qlib研究增强')).toBeInTheDocument();
    expect(screen.getByText('生成 Qlib 选股模型解释报告')).toBeInTheDocument();
    expect(screen.getByLabelText('生成 Qlib 选股模型解释报告 TopK')).toBeInTheDocument();
    expect(screen.getByText('生成 Qlib 回测明细报告')).toBeInTheDocument();
    expect(screen.getByText('生成 Qlib 多选股模型/多实验对比')).toBeInTheDocument();
    expect(screen.getByLabelText('生成 Qlib 多选股模型/多实验对比 对比 Run 数')).toBeInTheDocument();
    expect(screen.getByText('生成 Qlib 因子分析报告')).toBeInTheDocument();
    expect(screen.getByText('生成 Qlib 滚动训练计划')).toBeInTheDocument();
    expect(screen.getByLabelText('生成 Qlib 滚动训练计划 窗口数')).toBeInTheDocument();
    expect(screen.getByText('生成 Qlib 组合风险诊断')).toBeInTheDocument();
    expect(screen.getByText('生成 china-astock 策略对比')).toBeInTheDocument();
    expect(screen.getByText('读取 china-astock-quant 既有搜索和赢家评估结果，生成策略对比 CSV/Markdown 报告；只读报告，不连接券商或交易接口。')).toBeInTheDocument();
    expect(screen.getByText('生成 china-astock 技术指标报告')).toBeInTheDocument();
    expect(screen.getByText('复用 china-astock-quant 的 SMA/EMA/RSI/ATR/Bollinger 指标函数，读取 Qlib Provider 前复权行情生成只读技术面观察报告，不连接券商、不自动下单。')).toBeInTheDocument();
    expect(screen.getByLabelText('生成 china-astock 技术指标报告 股票代码')).toBeInTheDocument();
    expect(screen.getByText('生成数据治理盘点和清理预案')).toBeInTheDocument();
    expect(screen.getByTestId('workflow-job-report_data_hygiene')).toBeInTheDocument();
    expect(screen.getByTestId('workflow-run-report_data_hygiene')).toHaveAccessibleName('运行 生成数据治理盘点和清理预案');
    expect(screen.getByText('只读盘点当前数据、日志、旧 provider 和临时脚本，生成 inventory 与 dry-run 清理计划；不移动、不删除、不改配置。')).toBeInTheDocument();
    expect(screen.getByText('生成工作台能力盘点')).toBeInTheDocument();
    expect(screen.getByText('只读扫描 DSA 页面/API、quant_workflow 脚本、Qlib 接入和 china-astock-quant 可复用能力，输出功能入口和缺口清单。')).toBeInTheDocument();
    expect(screen.getByText('今日流程')).toBeInTheDocument();
    expect(screen.getByText('按顺序完成检查、报告、预演、训练和记录。')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '今日流程 异常清单 异常清单' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '今日流程 清洗预演 预演' })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: '实验记录' })).toHaveAttribute('href', '/research');
    expect(document.body).not.toHaveTextContent('原生 V2');
    expect(screen.getByRole('link', { name: '回测页' })).toHaveAttribute('href', '/backtest');
    expect(screen.getByText('检查 Qlib 数据')).toBeInTheDocument();
    expect(screen.getAllByText('训练 Qlib LightGBM 选股模型基线')).toHaveLength(2);
    expect(screen.getByText('latest_training_result.json')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'latest_training_result.json' })).toHaveAttribute(
      'href',
      '/api/v1/quant/jobs/qj_train/artifacts/0',
    );
  });

  it('shows a clear unavailable MLflow state from the quant status API', async () => {
    getQuantStatus.mockResolvedValueOnce({
      availableJobTypes: {},
      activeCount: 0,
      latestJobs: {},
      recentJobs: [],
      mlflow: {
        url: 'http://127.0.0.1:6006/',
        status: 'unavailable',
        role: 'experiment_raw_records',
        checkedAt: '2026-07-03T13:10:00Z',
        error: 'connection refused',
      },
    });

    renderWorkflowPage();

    expect(await screen.findByText('MLflow 未启动')).toBeInTheDocument();
    expect(screen.getAllByText('未启动').length).toBeGreaterThan(0);
    expect(screen.getByText('connection refused')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: '打开 MLflow，新窗口' })).toHaveAttribute('href', '/experiments/');
  });

  it('warns when the candidate pipeline is from the old compatibility provider', async () => {
    getResearchStatus.mockResolvedValueOnce({
      ...readyResearchStatus(),
      candidatePipeline: {
        status: 'completed',
        providerStatus: 'stale',
        providerCurrent: false,
        providerWarning: '该候选报告使用旧兼容数据，只能作为历史参考。',
        models: [
          {
            id: 'xgboost',
            label: 'XGBoost',
            status: 'completed',
            runId: 'run-xgb',
          },
        ],
        strategyEvaluations: [
          {
            modelId: 'xgboost',
            strategyId: 'combined_risk',
            strategyLabel: '完整组合风控',
          },
        ],
      },
    });

    renderWorkflowPage('training');

    expect(
      await screen.findByText('该候选报告使用旧兼容数据，只能作为历史参考。'),
    ).toBeInTheDocument();
  });

  it('warns separately when the latest training result is historical', async () => {
    getResearchStatus.mockResolvedValueOnce({
      ...readyResearchStatus(),
      latestTraining: {
        status: 'completed',
        experimentName: 'baseline_lgb_alpha158',
        runId: 'abc123',
        providerStatus: 'stale',
        providerCurrent: false,
        providerWarning: '该训练结果使用旧数据，只能作为历史参考。',
      },
    });

    renderWorkflowPage('training');

    expect(
      await screen.findByText('该训练结果使用旧数据，只能作为历史参考。'),
    ).toBeInTheDocument();
    expect(screen.getByText('训练结果来自旧数据')).toBeInTheDocument();
  });

  it('labels an invalid native catalog distinctly in English', async () => {
    localStorage.setItem(UI_LANGUAGE_STORAGE_KEY, 'en');
    getResearchStatus.mockResolvedValueOnce({
      ...readyResearchStatus(),
      candidatePipeline: {
        status: 'completed',
        providerStatus: 'catalog_invalid',
        providerCurrent: false,
        providerWarning: '后端中文诊断不应直接显示',
        models: [],
        strategyEvaluations: [],
      },
    });

    render(
      <UiLanguageProvider>
        <MemoryRouter>
          <QuantTaskPage surface="training" />
        </MemoryRouter>
      </UiLanguageProvider>,
    );

    expect(await screen.findByText('Qlib Provider identity invalid')).toBeInTheDocument();
    expect(screen.getByText(/current canonical dataset cannot be verified/i)).toBeInTheDocument();
    expect(screen.queryByText(/old compatibility data/i)).not.toBeInTheDocument();
    expect(document.body).not.toHaveTextContent(/native V2/i);
  });

  it('localizes the MLflow status card in English mode', async () => {
    localStorage.setItem(UI_LANGUAGE_STORAGE_KEY, 'en');
    getQuantStatus.mockResolvedValueOnce({
      availableJobTypes: {},
      activeCount: 0,
      latestJobs: {},
      recentJobs: [],
      mlflow: {
        url: 'https://mlflow.example.com/',
        status: 'linked',
        role: 'experiment_raw_records',
        checkedAt: null,
        error: 'Health check skipped for non-local MLflow URL.',
      },
    });

    render(
      <UiLanguageProvider>
        <MemoryRouter>
          <QuantTaskPage surface="all" />
        </MemoryRouter>
      </UiLanguageProvider>,
    );

    expect(await screen.findByText('MLflow External Records')).toBeInTheDocument();
    expect(screen.getByText('External Link')).toBeInTheDocument();
    expect(screen.queryByText('外部链接')).not.toBeInTheDocument();
    expect(screen.getByText('Health check skipped for non-local MLflow URL.')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Open MLflow in a new window' })).toHaveAttribute('href', 'https://mlflow.example.com/');
  });

  it('submits a whitelisted quant job from the workflow page', async () => {
    renderWorkflowPage();

    fireEvent.click(await screen.findByRole('button', { name: '运行 检查 Qlib 数据' }));

    await waitFor(() => {
      expect(submitQuantJob).toHaveBeenCalledWith('check_qlib_data', {});
    });

    expect(await screen.findByText('qj_check')).toBeInTheDocument();
  });

  it('does not increment the active count when a submitted job already failed to start', async () => {
    submitQuantJob.mockResolvedValueOnce({
      jobId: 'qj_failed_start',
      jobType: 'check_qlib_data',
      label: '检查 Qlib 数据',
      status: 'failed',
      params: {},
      command: [],
      cwd: '/Users/cq/WorkingProjects',
      createdAt: '2026-07-03T22:00:00+08:00',
      completedAt: '2026-07-03T22:00:01+08:00',
      updatedAt: '2026-07-03T22:00:01+08:00',
      error: 'Failed to start job: boom',
      logPath: '/tmp/job.log',
      artifactPaths: [],
    });
    renderWorkflowPage();

    expect(await screen.findByText('0 个运行中')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: '运行 检查 Qlib 数据' }));

    await waitFor(() => {
      expect(submitQuantJob).toHaveBeenCalledWith('check_qlib_data', {});
    });
    expect(await screen.findByText('qj_failed_start')).toBeInTheDocument();
    expect(screen.getByText('0 个运行中')).toBeInTheDocument();
    expect(screen.queryByText('1 个运行中')).not.toBeInTheDocument();
  });

  it('shows frontend and backend builds when the quant workbench contract matches', async () => {
    renderWorkflowPage('training');

    expect(await screen.findByText(/前端构建/)).toBeInTheDocument();
    expect(screen.getByText(/后端构建 backend-test-123/)).toBeInTheDocument();
    expect(screen.queryByText('网页与后端版本不兼容')).not.toBeInTheDocument();
  });

  it('explains an old backend contract and keeps run actions disabled', async () => {
    getQuantJobTypes.mockResolvedValueOnce({
      availableJobTypes: {
        train_baseline_lgb: {
          label: '训练 Qlib LightGBM 选股模型基线',
          allowedParams: ['config', 'experiment_name', 'provider'],
        },
      },
      apiContract: {
        version: 'quant-workbench-v1',
        backendBuildId: 'backend-old',
        backendStartedAt: '2026-08-01T01:00:00Z',
      },
    });

    renderWorkflowPage('training');

    expect(await screen.findByText('网页与后端版本不兼容')).toBeInTheDocument();
    expect(screen.getByText(/后端版本过旧或尚未重启/)).toBeInTheDocument();
    expect(screen.getByTestId('workflow-run-train_baseline_lgb')).toBeDisabled();
  });

  it('shows the Provider calendar cutoff without using historical report freshness', async () => {
    renderWorkflowPage('training');

    const baselineCard = await screen.findByTestId('workflow-job-train_baseline_lgb');
    expect(within(baselineCard).getByText(/实时检查：通过/)).toBeInTheDocument();
    expect(within(baselineCard).getByText(/Provider 日历截止 2026-07-24/)).toBeInTheDocument();
    expect(within(baselineCard).queryByText(/已落后 9 天/)).not.toBeInTheDocument();
    expect(screen.getByTestId('workflow-run-train_baseline_lgb')).toBeEnabled();
  });

  it('uses lightweight training preflight instead of previous quality evidence', async () => {
    const status = readyResearchStatus();
    getResearchStatus.mockResolvedValue({
      ...status,
      trainable: true,
      trainingReadiness: {
        status: 'trainable',
        summary: 'quant_data 已发布所需数据集，当前 Qlib Provider 轻量预检通过。',
      },
      trainingPreflight: {
        status: 'passed',
        requiredDatasets: ['ashare_daily_qfq', 'cn_index_daily', 'pit_csi800'],
        availableDatasets: [
          'ashare_daily_qfq',
          'cn_index_daily',
          'pit_csi800',
          'qlib_cleaning_plan',
        ],
        summary: 'quant_data 已发布所需数据集，当前 Qlib Provider 轻量预检通过。',
      },
      currentProviderIdentities: {
        ...status.currentProviderIdentities,
        canonical: {
          provider: 'canonical',
          datasetId: 'ashare_daily_qfq',
          snapshotId: 'ashare-baostock-qfq-20260805-new',
          calendarEnd: '2026-08-04',
          status: 'ready',
        },
        cleaned: {
          provider: 'cleaned',
          datasetId: 'ashare_daily_qfq',
          snapshotId: 'ashare-baostock-qfq-20260805-new',
          calendarEnd: '2026-08-04',
          status: 'ready',
        },
      },
    });

    renderWorkflowPage('training');

    expect(await screen.findByText('实时检查通过')).toBeInTheDocument();
    expect(screen.getByText(/quant_data 已发布所需数据集/)).toBeInTheDocument();
    expect(screen.queryByText('新快照待验收')).not.toBeInTheDocument();
    expect(screen.queryByText(/数据验收/)).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: '运行正式全量数据验收' })).not.toBeInTheDocument();
    expect(screen.queryByText(/重新验收数据/)).not.toBeInTheDocument();
    expect(screen.queryByText(/上次已验收快照/)).not.toBeInTheDocument();
    const baselineCard = screen.getByTestId('workflow-job-train_baseline_lgb');
    expect(within(baselineCard).getByText(/快照 ashare-baostock-qfq-20260805-new/)).toBeInTheDocument();
    expect(screen.getByTestId('workflow-run-train_baseline_lgb')).toBeEnabled();
    fireEvent.change(
      within(baselineCard).getByLabelText('训练 Qlib LightGBM 选股模型基线 数据版本'),
      { target: { value: 'cleaned' } },
    );
    expect(screen.getByTestId('workflow-run-train_baseline_lgb')).toBeEnabled();
  });

  it('reports mixed Provider preflight without consulting historical quality evidence', async () => {
    const status = readyResearchStatus();
    getResearchStatus.mockResolvedValue({
      ...status,
      trainable: true,
      trainingPreflights: {
        canonical: status.trainingPreflights.canonical,
        cleaned: {
          ...status.trainingPreflights.cleaned,
          status: 'blocked',
          summary: '当前 cleaned Qlib Provider 缺少完整的不可变数据血缘，正式训练已阻止。',
        },
      },
      currentProviderIdentities: {
        canonical: {
          provider: 'canonical',
          datasetId: 'qlib_ashare_v2',
          snapshotId: 'snapshot-20260802',
          calendarEnd: '2026-07-24',
          status: 'ready',
        },
        cleaned: {
          provider: 'cleaned',
          datasetId: 'qlib_ashare_cleaned_v2',
          snapshotId: 'cleaned-snapshot-20260805-new',
          calendarEnd: '2026-08-04',
          status: 'ready',
        },
      },
    });

    renderWorkflowPage('training');

    expect(await screen.findByText('部分 Provider 实时检查通过')).toBeInTheDocument();
    expect(screen.getByText('实时检查：部分通过，部分阻塞')).toBeInTheDocument();
    expect(screen.getByText(/canonical 实时检查通过；cleaned 当前不能用于正式训练/)).toBeInTheDocument();
    expect(screen.getByTestId('workflow-run-train_baseline_lgb')).toBeEnabled();

    const baselineCard = screen.getByTestId('workflow-job-train_baseline_lgb');
    fireEvent.change(
      within(baselineCard).getByLabelText('训练 Qlib LightGBM 选股模型基线 数据版本'),
      { target: { value: 'cleaned' } },
    );

    const selectedProviderStatus = within(baselineCard).getByText(/当前数据：cleaned/);
    expect(selectedProviderStatus).toHaveTextContent('实时检查未通过');
    expect(selectedProviderStatus).toHaveTextContent('缺少完整的不可变数据血缘');
    expect(screen.getByTestId('workflow-run-train_baseline_lgb')).toBeDisabled();
  });

  it('does not present the previous accepted snapshot as current when the Qlib Provider is unavailable', async () => {
    const status = readyResearchStatus();
    getResearchStatus.mockResolvedValue({
      ...status,
      trainable: false,
      trainingReadiness: {
        status: 'blocked',
        summary: 'The current Qlib Provider is unavailable.',
      },
      trainingPreflights: {
        canonical: {
          status: 'blocked',
          provider: 'canonical',
          summary: 'The current Qlib Provider is unavailable.',
        },
        cleaned: status.trainingPreflights.cleaned,
      },
      currentProviderIdentities: {
        canonical: {
          provider: 'canonical',
          datasetId: 'qlib_ashare_v2',
          status: 'blocked',
          error: 'Current provider is unavailable.',
        },
      },
    });

    renderWorkflowPage('training');

    expect(await screen.findByText('数据暂不可训练')).toBeInTheDocument();
    expect(screen.getByText('实时检查：阻塞')).toBeInTheDocument();
    const baselineCard = screen.getByTestId('workflow-job-train_baseline_lgb');
    expect(within(baselineCard).getByText(/快照 -/)).toBeInTheDocument();
    expect(within(baselineCard).getByText(/实时检查未通过/)).toBeInTheDocument();
    expect(baselineCard).not.toHaveTextContent('上次已验收快照');
    expect(baselineCard).not.toHaveTextContent('快照 snapshot-20260802');
    expect(screen.getByTestId('workflow-run-train_baseline_lgb')).toBeDisabled();
  });

  it('localizes the live data-readiness summary without changing the API payload', async () => {
    const status = readyResearchStatus();
    const rawSummary = 'Qlib provider has enough calendar/instrument coverage for the baseline run. Warnings: 3152 symbols have non-positive volume rows; 252 symbols have absolute daily returns above 50%';
    status.trainingReadiness.summary = rawSummary;
    getResearchStatus.mockResolvedValue(status);

    renderWorkflowPage('training');

    expect(await screen.findByText(/Qlib 数据快照的交易日和股票覆盖已达到基线训练要求/)).toBeInTheDocument();
    expect(screen.getByText(/3152 只股票存在非正成交量记录/)).toBeInTheDocument();
    expect(screen.getByText(/252 只股票存在单日绝对涨跌幅超过 50% 的记录/)).toBeInTheDocument();
    expect(screen.queryByText(/Qlib provider has enough calendar/i)).not.toBeInTheDocument();
    expect(status.trainingReadiness.summary).toBe(rawSummary);
  });

  it('disables workflow jobs that are not advertised by the backend whitelist', async () => {
    getQuantJobTypes.mockResolvedValueOnce({
      availableJobTypes: {
        check_qlib_data: { label: '检查 Qlib 数据', allowedParams: [] },
      },
      apiContract: {
        version: 'quant-workbench-v2',
        backendBuildId: 'backend-test-123',
        backendStartedAt: '2026-08-02T02:00:00Z',
      },
    });
    getQuantStatus.mockResolvedValueOnce({
      availableJobTypes: {
        check_qlib_data: { label: '检查 Qlib 数据', allowedParams: [] },
      },
      activeCount: 0,
      latestJobs: {},
      recentJobs: [],
    });

    renderWorkflowPage();

    const hygieneButton = await screen.findByRole('button', { name: '运行 生成数据治理盘点和清理预案' });
    expect(hygieneButton).toBeDisabled();
    expect(screen.getAllByText('后端未启用').length).toBeGreaterThan(0);

    fireEvent.click(hygieneButton);

    expect(submitQuantJob).not.toHaveBeenCalledWith('report_data_hygiene', expect.anything());
  });

  it('waits for the backend whitelist before reporting a job as disabled', async () => {
    let resolveStatus: (value: unknown) => void = () => undefined;
    getQuantJobTypes.mockReturnValueOnce(new Promise((resolve) => {
      resolveStatus = resolve;
    }));
    getQuantStatus.mockReturnValueOnce(new Promise((resolve) => {
      void resolve;
    }));

    renderWorkflowPage();

    const hygieneButton = await screen.findByRole('button', { name: '运行 生成数据治理盘点和清理预案' });
    const hygieneHeading = screen.getByText('生成数据治理盘点和清理预案');
    const hygieneHeadingCluster = hygieneHeading.closest('div');
    expect(hygieneButton).toBeDisabled();
    expect(screen.queryByText('后端未启用')).not.toBeInTheDocument();

    await act(async () => {
      resolveStatus({
        availableJobTypes: {
          report_data_hygiene: { label: '生成数据治理盘点和清理预案', allowedParams: [] },
        },
        apiContract: {
          version: 'quant-workbench-v2',
          backendBuildId: 'backend-test-123',
          backendStartedAt: '2026-08-02T02:00:00Z',
        },
      });
    });

    await waitFor(() => {
      expect(hygieneButton).not.toBeDisabled();
    });
    expect(within(hygieneHeadingCluster as HTMLElement).queryByText('后端未启用')).not.toBeInTheDocument();
  });

  it('keeps daily guide job shortcuts disabled until the backend whitelist loads', async () => {
    let resolveStatus: (value: unknown) => void = () => undefined;
    getQuantJobTypes.mockReturnValueOnce(new Promise((resolve) => {
      resolveStatus = resolve;
    }));
    getQuantStatus.mockReturnValueOnce(new Promise((resolve) => {
      void resolve;
    }));

    renderWorkflowPage();

    const guideButton = await screen.findByRole('button', { name: '今日流程 清洗预演 预演' });
    expect(guideButton).toBeDisabled();

    fireEvent.click(guideButton);
    expect(submitQuantJob).not.toHaveBeenCalled();
    expect(screen.queryByText('后端未启用')).not.toBeInTheDocument();

    await act(async () => {
      resolveStatus({
        availableJobTypes: {
          apply_data_cleaning: { label: '生成正式 cleaned Qlib Provider', allowedParams: ['dry_run'] },
        },
        apiContract: {
          version: 'quant-workbench-v2',
          backendBuildId: 'backend-test-123',
          backendStartedAt: '2026-08-02T02:00:00Z',
        },
      });
    });

    await waitFor(() => {
      expect(guideButton).not.toBeDisabled();
    });
  });

  it('disables daily guide job shortcuts that are missing from the backend whitelist', async () => {
    getQuantJobTypes.mockResolvedValueOnce({
      availableJobTypes: {
        check_qlib_data: { label: '检查 Qlib 数据', allowedParams: [] },
      },
      apiContract: {
        version: 'quant-workbench-v2',
        backendBuildId: 'backend-test-123',
        backendStartedAt: '2026-08-02T02:00:00Z',
      },
    });
    getQuantStatus.mockResolvedValueOnce({
      availableJobTypes: {
        check_qlib_data: { label: '检查 Qlib 数据', allowedParams: [] },
      },
      activeCount: 0,
      latestJobs: {},
      recentJobs: [],
    });

    renderWorkflowPage();

    const guideButton = await screen.findByRole('button', { name: '今日流程 清洗预演 预演' });
    await waitFor(() => {
      expect(guideButton).toBeDisabled();
    });
    expect(screen.getAllByText('后端未启用').length).toBeGreaterThan(0);

    fireEvent.click(guideButton);
    expect(submitQuantJob).not.toHaveBeenCalledWith('apply_data_cleaning', expect.anything());
  });

  it('submits provider maintenance preview jobs without user params', async () => {
    renderWorkflowPage();

    fireEvent.click(await screen.findByRole('button', { name: '运行 预演旧兼容 CSI300 股票池' }));
    fireEvent.click(await screen.findByRole('button', { name: '运行 预演旧兼容 benchmark' }));

    await waitFor(() => {
      expect(submitQuantJob).toHaveBeenCalledWith('generate_csi300', {});
      expect(submitQuantJob).toHaveBeenCalledWith('add_index_benchmark', {});
    });
  });

  it('submits the read-only workbench capability inventory job without parameters', async () => {
    renderWorkflowPage();

    fireEvent.click(await screen.findByRole('button', { name: '运行 生成工作台能力盘点' }));

    await waitFor(() => {
      expect(submitQuantJob).toHaveBeenCalledWith('report_workbench_capabilities', {});
    });
  });

  it('enables jobs from the fast whitelist while the full quant status is still loading', async () => {
    getQuantStatus.mockReturnValueOnce(new Promise(() => undefined));
    getResearchStatus.mockReturnValueOnce(new Promise(() => undefined));
    listQuantJobs.mockReturnValueOnce(new Promise(() => undefined));

    renderWorkflowPage('other');

    const button = await screen.findByRole('button', { name: '运行 生成工作台能力盘点' });
    await waitFor(() => expect(button).not.toBeDisabled());
  });

  it('submits Qlib research enhancement jobs with fixed whitelist params', async () => {
    renderWorkflowPage();

    const explainTopk = await screen.findByLabelText('生成 Qlib 选股模型解释报告 TopK');
    fireEvent.change(explainTopk, { target: { value: '12' } });
    fireEvent.click(screen.getByRole('button', { name: '运行 生成 Qlib 选股模型解释报告' }));

    const rollingWindows = screen.getByLabelText('生成 Qlib 滚动训练计划 窗口数');
    fireEvent.change(rollingWindows, { target: { value: '4' } });
    fireEvent.click(screen.getByRole('button', { name: '运行 生成 Qlib 滚动训练计划' }));

    await waitFor(() => {
      expect(submitQuantJob).toHaveBeenCalledWith('report_qlib_model_explainability', { topk: '12' });
      expect(submitQuantJob).toHaveBeenCalledWith('report_qlib_rolling_training', {
        windows: '4',
        train_years: '3',
        valid_months: '12',
        test_months: '1',
        step_months: '1',
      });
    });
  });

  it('prevents duplicate submissions for the same running workflow action', async () => {
    let resolveSubmit: (value: unknown) => void = () => undefined;
    submitQuantJob.mockImplementationOnce(() => new Promise((resolve) => {
      resolveSubmit = resolve;
    }));

    renderWorkflowPage();

    const runButton = await screen.findByRole('button', { name: '运行 检查 Qlib 数据' });
    fireEvent.click(runButton);
    fireEvent.click(runButton);

    expect(submitQuantJob).toHaveBeenCalledTimes(1);

    resolveSubmit({
      jobId: 'qj_check_dedup',
      jobType: 'check_qlib_data',
      label: '检查 Qlib 数据',
      status: 'running',
      params: {},
      command: [],
      cwd: '/Users/cq/WorkingProjects',
      createdAt: '2026-07-03T22:00:00+08:00',
      updatedAt: '2026-07-03T22:00:00+08:00',
      logPath: '/tmp/job.log',
      artifactPaths: [],
    });
    expect(await screen.findByText('qj_check_dedup')).toBeInTheDocument();
  });

  it('shows quant job submission errors in the workflow alert', async () => {
    submitQuantJob.mockRejectedValueOnce(createParsedApiError({
      title: '提交失败',
      message: '后端拒绝任务',
      rawMessage: '后端拒绝任务',
    }));

    renderWorkflowPage();

    fireEvent.click(await screen.findByRole('button', { name: '运行 检查 Qlib 数据' }));

    expect(await screen.findByText('工作流接口异常')).toBeInTheDocument();
    expect(screen.getByText('后端拒绝任务')).toBeInTheDocument();
  });

  it('submits browser-provided parameters for jobs that support them', async () => {
    renderWorkflowPage();

    fireEvent.change(await screen.findByLabelText('运行 DSA 分析准确性回测 股票代码'), {
      target: { value: '600519' },
    });
    fireEvent.change(screen.getByLabelText('运行 DSA 分析准确性回测 评估窗口'), {
      target: { value: '1' },
    });
    fireEvent.click(screen.getByLabelText('运行 DSA 分析准确性回测 强制重跑'));
    fireEvent.click(screen.getByRole('button', { name: '运行 运行 DSA 分析准确性回测' }));

    await waitFor(() => {
      expect(submitQuantJob).toHaveBeenCalledWith('run_dsa_backtest', {
        code: '600519',
        eval_window_days: '1',
        force: true,
      });
    });
  });

  it('submits conservative defaults for an A-share network source job', async () => {
    renderWorkflowPage();

    fireEvent.click(await screen.findByRole('button', { name: '运行 下载东方财富日线' }));

    await waitFor(() => {
      expect(submitQuantJob).toHaveBeenCalledWith('download_eastmoney', {
        incremental: true,
        sleep: '2',
        retries: '1',
        max_consecutive_failures: '3',
        max_requests: '6000',
        overlap_days: '7',
      });
    });
  });

  it('submits the selected Qlib provider for training jobs', async () => {
    const status = readyResearchStatus();
    getResearchStatus.mockResolvedValueOnce({
      ...status,
      providerQuality: {
        ...status.providerQuality,
        cleaned: {
          provider: 'cleaned',
          datasetId: 'qlib_ashare_cleaned_v2',
          snapshotId: 'cleaned-snapshot-ready',
          trainable: true,
          trainingReadiness: { status: 'trainable', summary: 'cleaned ready' },
        },
      },
    });
    renderWorkflowPage();

    fireEvent.change(await screen.findByLabelText('训练 Qlib LightGBM 选股模型基线 数据版本'), {
      target: { value: 'cleaned' },
    });
    fireEvent.click(screen.getByRole('button', { name: '运行 训练 Qlib LightGBM 选股模型基线' }));

    await waitFor(() => {
      expect(submitQuantJob).toHaveBeenCalledWith('train_baseline_lgb', {
        provider: 'cleaned',
      });
    });
  });

  it('renders and submits only parameters advertised by the backend whitelist', async () => {
    getQuantJobTypes.mockResolvedValueOnce({
      availableJobTypes: {
        train_baseline_lgb: { label: '训练 Qlib LightGBM 选股模型基线', allowedParams: ['config'] },
      },
      apiContract: {
        version: 'quant-workbench-v2',
        backendBuildId: 'backend-test-123',
        backendStartedAt: '2026-08-02T02:00:00Z',
      },
    });
    getQuantStatus.mockResolvedValueOnce({
      availableJobTypes: {
        train_baseline_lgb: { label: '训练 Qlib LightGBM 选股模型基线', allowedParams: ['config'] },
      },
      activeCount: 0,
      latestJobs: {},
      recentJobs: [],
    });

    renderWorkflowPage();

    expect(await screen.findByLabelText('训练 Qlib LightGBM 选股模型基线 配置文件')).toBeInTheDocument();
    expect(screen.queryByLabelText('训练 Qlib LightGBM 选股模型基线 数据版本')).not.toBeInTheDocument();
    expect(screen.queryByLabelText('训练 Qlib LightGBM 选股模型基线 实验名称')).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: '运行 训练 Qlib LightGBM 选股模型基线' }));

    await waitFor(() => {
      expect(submitQuantJob).toHaveBeenCalledWith('train_baseline_lgb', {});
    });
  });

  it('submits the isolated smoke fixture flag for Qlib merge jobs', async () => {
    renderWorkflowPage();

    fireEvent.click(await screen.findByLabelText('生成正式 Qlib Provider 使用隔离小样本演练'));
    fireEvent.click(screen.getByRole('button', { name: '运行 生成正式 Qlib Provider' }));

    await waitFor(() => {
      expect(submitQuantJob).toHaveBeenCalledWith('merge_to_qlib', {
        smoke: true,
      });
    });
  });

  it('submits JQData browser-safe dry-run parameters from the workflow page', async () => {
    renderWorkflowPage();

    fireEvent.change(await screen.findByLabelText('验证 JQData 连接 股票池'), {
      target: { value: 'csi300' },
    });
    fireEvent.change(screen.getByLabelText('验证 JQData 连接 验证样本数'), {
      target: { value: '1' },
    });
    fireEvent.change(screen.getByLabelText('验证 JQData 连接 请求间隔秒'), {
      target: { value: '0' },
    });
    fireEvent.click(screen.getByRole('button', { name: '运行 验证 JQData 连接' }));

    await waitFor(() => {
      expect(submitQuantJob).toHaveBeenCalledWith('download_jqdata', {
        dry_run: true,
        universe: 'csi300',
        limit: '1',
        sleep: '0',
      });
    });
  });

  it('submits market regime report parameters from the workflow page', async () => {
    renderWorkflowPage();

    fireEvent.change(await screen.findByLabelText('生成市场状态和现金仓位信号 回看交易日'), {
      target: { value: '120' },
    });
    fireEvent.change(screen.getByLabelText('生成市场状态和现金仓位信号 最少股票数'), {
      target: { value: '50' },
    });
    fireEvent.change(screen.getByLabelText('生成市场状态和现金仓位信号 抽样上限'), {
      target: { value: '200' },
    });
    fireEvent.click(screen.getByRole('button', { name: '运行 生成市场状态和现金仓位信号' }));

    await waitFor(() => {
      expect(submitQuantJob).toHaveBeenCalledWith('report_market_regime', {
        lookback_days: '120',
        min_symbols: '50',
        limit: '200',
      });
    });
  });

  it('submits cash-control backtest parameters from the workflow page', async () => {
    renderWorkflowPage();

    fireEvent.change(await screen.findByLabelText('生成现金仓位叠加回测 回看交易日'), {
      target: { value: '80' },
    });
    fireEvent.change(screen.getByLabelText('生成现金仓位叠加回测 最少股票数'), {
      target: { value: '2' },
    });
    fireEvent.change(screen.getByLabelText('生成现金仓位叠加回测 抽样上限'), {
      target: { value: '20' },
    });
    fireEvent.click(screen.getByRole('button', { name: '运行 生成现金仓位叠加回测' }));

    await waitFor(() => {
      expect(submitQuantJob).toHaveBeenCalledWith('report_cash_control_backtest', {
        lookback_days: '80',
        min_symbols: '2',
        limit: '20',
      });
    });
  });

  it('submits Qlib prediction summary parameters from the workflow page', async () => {
    renderWorkflowPage();

    fireEvent.change(await screen.findByLabelText('生成 Qlib 预测摘要 TopK'), {
      target: { value: '15' },
    });
    fireEvent.change(screen.getByLabelText('生成 Qlib 预测摘要 MLflow Run ID'), {
      target: { value: 'safe_run_20260705' },
    });
    fireEvent.click(screen.getByRole('button', { name: '运行 生成 Qlib 预测摘要' }));

    await waitFor(() => {
      expect(submitQuantJob).toHaveBeenCalledWith('report_qlib_predictions', {
        topk: '15',
        run_id: 'safe_run_20260705',
      });
    });
  });

  it('submits the cleaned Qlib provider task from the workflow page', async () => {
    renderWorkflowPage();

    fireEvent.click(await screen.findByLabelText('生成正式 cleaned Qlib Provider 只预演不写入'));
    fireEvent.click(screen.getByRole('button', { name: '运行 生成正式 cleaned Qlib Provider' }));

    await waitFor(() => {
      expect(submitQuantJob).toHaveBeenCalledWith('apply_data_cleaning', {
        dry_run: true,
      });
    });
  });

  it('runs the daily guide cleaning preflight with safe dry-run parameters', async () => {
    renderWorkflowPage();

    fireEvent.click(await screen.findByRole('button', { name: '今日流程 清洗预演 预演' }));

    await waitFor(() => {
      expect(submitQuantJob).toHaveBeenCalledWith('apply_data_cleaning', {
        dry_run: true,
      });
    });
  });

  it('submits the china-astock report-only comparison job without parameters', async () => {
    renderWorkflowPage();

    fireEvent.click(await screen.findByRole('button', { name: '运行 生成 china-astock 策略对比' }));

    await waitFor(() => {
      expect(submitQuantJob).toHaveBeenCalledWith('run_china_astock_comparison', {});
    });
  });

  it('submits the china-astock indicator report parameters from the workflow page', async () => {
    renderWorkflowPage();

    fireEvent.change(await screen.findByLabelText('生成 china-astock 技术指标报告 股票代码'), {
      target: { value: '000001,600519' },
    });
    fireEvent.change(screen.getByLabelText('生成 china-astock 技术指标报告 回看交易日'), {
      target: { value: '60' },
    });
    fireEvent.change(screen.getByLabelText('生成 china-astock 技术指标报告 股票数量上限'), {
      target: { value: '2' },
    });
    fireEvent.click(screen.getByRole('button', { name: '运行 生成 china-astock 技术指标报告' }));

    await waitFor(() => {
      expect(submitQuantJob).toHaveBeenCalledWith('report_china_astock_indicators', {
        symbols: '000001,600519',
        lookback_days: '60',
        limit: '2',
      });
    });
  });

  it('submits an empty china-astock indicator report form so the backend applies safe defaults', async () => {
    renderWorkflowPage();

    fireEvent.click(await screen.findByRole('button', { name: '运行 生成 china-astock 技术指标报告' }));

    await waitFor(() => {
      expect(submitQuantJob).toHaveBeenCalledWith('report_china_astock_indicators', {});
    });
  });

  it('submits browser-safe strategy optimization dry-run parameters', async () => {
    renderWorkflowPage();

    fireEvent.change(await screen.findByLabelText('交易策略优化预演 阶段'), {
      target: { value: '4' },
    });
    fireEvent.change(screen.getByLabelText('交易策略优化预演 TopK'), {
      target: { value: '30' },
    });
    fireEvent.click(screen.getByRole('button', { name: '运行 交易策略优化预演' }));

    await waitFor(() => {
      expect(submitQuantJob).toHaveBeenCalledWith('optimize_strategy', {
        dry_run: true,
        phase: '4',
        topk: '30',
      });
    });
  });

  it('previews a completed text artifact inside the workflow page', async () => {
    renderWorkflowPage();

    fireEvent.click(await screen.findByRole('button', { name: '预览 latest_training_result.json' }));

    await waitFor(() => {
      expect(getQuantJobArtifactText).toHaveBeenCalledWith('qj_train', 0);
    });
    expect(await screen.findByRole('heading', { name: '产物预览' })).toBeInTheDocument();
    expect(screen.getByRole('log', { name: '产物预览 latest_training_result.json' })).toHaveTextContent('IC: 0.01');
    expect(screen.getAllByText('latest_training_result.json').length).toBeGreaterThan(0);
    expect(screen.getByText(/IC: 0.01/)).toBeInTheDocument();
  });

  it('keeps the newest artifact preview when an older request finishes later', async () => {
    const artifactJob = {
      jobId: 'qj_artifact_race',
      jobType: 'train_baseline_lgb',
      label: '训练 Qlib LightGBM 选股模型基线',
      status: 'completed',
      params: {},
      createdAt: '2026-08-02T10:00:00+08:00',
      completedAt: '2026-08-02T10:01:00+08:00',
      updatedAt: '2026-08-02T10:01:00+08:00',
      artifactPaths: ['/tmp/artifact-a.md', '/tmp/artifact-b.md'],
    };
    let resolveArtifactA: (value: string) => void = () => undefined;
    let resolveArtifactB: (value: string) => void = () => undefined;
    getQuantStatus.mockReturnValueOnce(new Promise(() => undefined));
    listQuantJobs.mockReturnValueOnce(new Promise(() => undefined));
    getQuantJobArtifactText.mockImplementation((_jobId: string, artifactIndex: number) => (
      new Promise<string>((resolve) => {
        if (artifactIndex === 0) resolveArtifactA = resolve;
        if (artifactIndex === 1) resolveArtifactB = resolve;
      })
    ));

    renderWorkflowPage('training');
    await waitFor(() => {
      expect(FakeEventSource.instances).toHaveLength(1);
    });
    act(() => {
      FakeEventSource.instances[0].emit('jobs', { jobs: [artifactJob] });
    });

    fireEvent.click(await screen.findByRole('button', { name: '预览 artifact-a.md' }));
    fireEvent.click(await screen.findByRole('button', { name: '预览 artifact-b.md' }));
    await act(async () => {
      resolveArtifactB('# Artifact B');
    });
    expect(screen.getByRole('log', { name: '产物预览 artifact-b.md' })).toHaveTextContent('Artifact B');

    await act(async () => {
      resolveArtifactA('# Artifact A');
    });
    expect(screen.getByRole('log', { name: '产物预览 artifact-b.md' })).toHaveTextContent('Artifact B');
    expect(screen.queryByText('Artifact A')).not.toBeInTheDocument();
  });

  it('shows a streamed artifact while the slower status snapshot is still loading', async () => {
    const completedJob = {
      jobId: 'qj_cash_control',
      jobType: 'report_cash_control_backtest',
      label: '生成现金仓位叠加回测',
      status: 'completed',
      params: {},
      createdAt: '2026-07-03T22:00:00+08:00',
      completedAt: '2026-07-03T22:00:01+08:00',
      updatedAt: '2026-07-03T22:00:01+08:00',
      artifactPaths: ['/tmp/cash_control_backtest_latest.md'],
    };
    getQuantStatus.mockReturnValueOnce(new Promise(() => undefined));
    listQuantJobs.mockReturnValueOnce(new Promise(() => undefined));

    renderWorkflowPage('other');

    await waitFor(() => {
      expect(FakeEventSource.instances).toHaveLength(1);
    });
    act(() => {
      FakeEventSource.instances[0].emit('jobs', { jobs: [completedJob] });
    });

    expect(
      await screen.findByRole('button', { name: '预览 cash_control_backtest_latest.md' }),
    ).toBeInTheDocument();
    expect(screen.getByText('qj_cash_control')).toBeInTheDocument();
  });

  it('does not let a delayed GET snapshot overwrite a newer streamed job', async () => {
    const staleRunningJob = {
      jobId: 'qj_refresh_race',
      jobType: 'check_qlib_data',
      label: '检查 Qlib 数据',
      status: 'running',
      params: {},
      createdAt: '2026-07-03T22:00:00+08:00',
      updatedAt: '2026-07-03T22:00:01+08:00',
      artifactPaths: ['/tmp/stale.log'],
    };
    const streamedCompletedJob = {
      ...staleRunningJob,
      status: 'completed',
      completedAt: '2026-07-03T22:01:00+08:00',
      updatedAt: '2026-07-03T22:01:00+08:00',
      artifactPaths: ['/tmp/final.log'],
    };
    let resolveStatus: (value: unknown) => void = () => undefined;
    let resolveList: (value: unknown) => void = () => undefined;
    getQuantStatus.mockReturnValueOnce(new Promise((resolve) => {
      resolveStatus = resolve;
    }));
    listQuantJobs.mockReturnValueOnce(new Promise((resolve) => {
      resolveList = resolve;
    }));

    renderWorkflowPage();
    await waitFor(() => {
      expect(FakeEventSource.instances).toHaveLength(1);
    });
    act(() => {
      FakeEventSource.instances[0].emit('jobs', { jobs: [streamedCompletedJob] });
    });
    expect(await screen.findByText('final.log')).toBeInTheDocument();

    await act(async () => {
      resolveStatus({
        availableJobTypes: {},
        activeCount: 1,
        latestJobs: { check_qlib_data: staleRunningJob },
        recentJobs: [staleRunningJob],
      });
      resolveList({ total: 1, items: [staleRunningJob] });
    });

    expect(screen.getByText('final.log')).toBeInTheDocument();
    expect(screen.queryByText('stale.log')).not.toBeInTheDocument();
    expect(screen.getByText('0 个运行中')).toBeInTheDocument();
    expect(screen.queryByText('1 个运行中')).not.toBeInTheDocument();
  });

  it('does not let a delayed status snapshot remove a newer streamed active job', async () => {
    const streamedRunningJob = {
      jobId: 'qj_created_after_status_request',
      jobType: 'check_qlib_data',
      label: '检查 Qlib 数据',
      status: 'running',
      params: {},
      createdAt: '2026-07-03T22:05:00+08:00',
      updatedAt: '2026-07-03T22:05:00+08:00',
      artifactPaths: [],
    };

    renderWorkflowPage();
    await screen.findByText('实时检查通过');
    expect(getQuantStatus).toHaveBeenCalledTimes(1);

    let resolveStatus: (value: unknown) => void = () => undefined;
    getQuantStatus.mockReturnValueOnce(new Promise((resolve) => {
      resolveStatus = resolve;
    }));
    fireEvent.click(screen.getByRole('button', { name: '刷新' }));
    await waitFor(() => {
      expect(getQuantStatus).toHaveBeenCalledTimes(2);
    });

    act(() => {
      FakeEventSource.instances[0].emit('jobs', { jobs: [streamedRunningJob] });
    });
    expect(await screen.findByText('qj_created_after_status_request')).toBeInTheDocument();
    expect(screen.getByText('1 个运行中')).toBeInTheDocument();

    await act(async () => {
      resolveStatus({
        availableJobTypes: {},
        activeCount: 0,
        latestJobs: {},
        recentJobs: [],
      });
    });

    expect(screen.getByText('qj_created_after_status_request')).toBeInTheDocument();
    expect(screen.getByText('1 个运行中')).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: '取消 检查 Qlib 数据' }),
    ).toBeInTheDocument();
  });

  it('does not let a delayed status snapshot resurrect a job removed by the stream', async () => {
    const staleRunningJob = {
      jobId: 'qj_completed_after_status_request',
      jobType: 'check_qlib_data',
      label: '检查 Qlib 数据',
      status: 'running',
      params: {},
      createdAt: '2026-07-03T22:00:00+08:00',
      updatedAt: '2026-07-03T22:00:00+08:00',
      artifactPaths: [],
    };
    getQuantStatus.mockResolvedValueOnce({
      availableJobTypes: {},
      activeCount: 1,
      latestJobs: { check_qlib_data: staleRunningJob },
      recentJobs: [staleRunningJob],
    });
    listQuantJobs.mockResolvedValueOnce({ total: 1, items: [staleRunningJob] });

    renderWorkflowPage();
    expect(await screen.findByText('qj_completed_after_status_request')).toBeInTheDocument();
    expect(screen.getByText('1 个运行中')).toBeInTheDocument();

    let resolveStatus: (value: unknown) => void = () => undefined;
    getQuantStatus.mockReturnValueOnce(new Promise((resolve) => {
      resolveStatus = resolve;
    }));
    fireEvent.click(screen.getByRole('button', { name: '刷新' }));
    await waitFor(() => {
      expect(getQuantStatus).toHaveBeenCalledTimes(2);
    });

    act(() => {
      FakeEventSource.instances[0].emit('jobs', { jobs: [] });
    });
    expect(await screen.findByText('0 个运行中')).toBeInTheDocument();
    expect(screen.queryByText('qj_completed_after_status_request')).not.toBeInTheDocument();

    await act(async () => {
      resolveStatus({
        availableJobTypes: {},
        activeCount: 1,
        latestJobs: { check_qlib_data: staleRunningJob },
        recentJobs: [staleRunningJob],
      });
    });

    expect(screen.getByText('0 个运行中')).toBeInTheDocument();
    expect(screen.queryByText('qj_completed_after_status_request')).not.toBeInTheDocument();
    expect(
      screen.queryByRole('button', { name: '取消 检查 Qlib 数据' }),
    ).not.toBeInTheDocument();
  });

  it('keeps the active count idempotent when SSE arrives before submit resolves', async () => {
    const submittedJob = {
      jobId: 'qj_submit_race',
      jobType: 'check_qlib_data',
      label: '检查 Qlib 数据',
      status: 'running',
      params: {},
      createdAt: '2026-07-03T22:00:00+08:00',
      updatedAt: '2026-07-03T22:00:00+08:00',
      artifactPaths: [],
    };
    let resolveSubmit: (value: unknown) => void = () => undefined;
    submitQuantJob.mockReturnValueOnce(new Promise((resolve) => {
      resolveSubmit = resolve;
    }));

    renderWorkflowPage();
    fireEvent.click(await screen.findByRole('button', { name: '运行 检查 Qlib 数据' }));
    await waitFor(() => {
      expect(submitQuantJob).toHaveBeenCalledWith('check_qlib_data', {});
    });
    act(() => {
      FakeEventSource.instances[0].emit('jobs', { jobs: [submittedJob] });
    });
    expect(await screen.findByText('1 个运行中')).toBeInTheDocument();

    await act(async () => {
      resolveSubmit(submittedJob);
    });

    expect(screen.getByText('1 个运行中')).toBeInTheDocument();
    expect(screen.queryByText('2 个运行中')).not.toBeInTheDocument();
  });

  it('does not decrement another active job when SSE precedes cancel response', async () => {
    const cancelledTarget = {
      jobId: 'qj_cancel_target',
      jobType: 'check_qlib_data',
      label: '检查 Qlib 数据',
      status: 'running',
      params: {},
      createdAt: '2026-07-03T22:00:00+08:00',
      updatedAt: '2026-07-03T22:00:00+08:00',
      artifactPaths: [],
    };
    const otherRunningJob = {
      jobId: 'qj_other_running',
      jobType: 'report_data_quality',
      label: '生成数据质量报告',
      status: 'running',
      params: {},
      createdAt: '2026-07-03T22:00:10+08:00',
      updatedAt: '2026-07-03T22:00:10+08:00',
      artifactPaths: [],
    };
    const cancelledResponse = {
      ...cancelledTarget,
      status: 'cancelled',
      completedAt: '2026-07-03T22:01:00+08:00',
      updatedAt: '2026-07-03T22:01:00+08:00',
    };
    getQuantStatus.mockResolvedValueOnce({
      availableJobTypes: {},
      activeCount: 2,
      latestJobs: {
        check_qlib_data: cancelledTarget,
        report_data_quality: otherRunningJob,
      },
      recentJobs: [otherRunningJob, cancelledTarget],
    });
    listQuantJobs.mockResolvedValueOnce({
      total: 2,
      items: [otherRunningJob, cancelledTarget],
    });
    let resolveCancel: (value: unknown) => void = () => undefined;
    cancelQuantJob.mockReturnValueOnce(new Promise((resolve) => {
      resolveCancel = resolve;
    }));

    renderWorkflowPage();
    expect(await screen.findByText('2 个运行中')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: '取消 检查 Qlib 数据' }));
    expect(cancelQuantJob).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole('button', { name: '确认取消' }));
    await waitFor(() => {
      expect(cancelQuantJob).toHaveBeenCalledWith('qj_cancel_target');
    });
    act(() => {
      FakeEventSource.instances[0].emit('jobs', {
        jobs: [otherRunningJob, cancelledResponse],
      });
    });
    expect(await screen.findByText('1 个运行中')).toBeInTheDocument();

    await act(async () => {
      resolveCancel(cancelledResponse);
    });

    expect(screen.getByText('1 个运行中')).toBeInTheDocument();
    expect(screen.queryByText('0 个运行中')).not.toBeInTheDocument();
  });

  it('retains every active job in addition to the bounded completed history', async () => {
    const longRunningJob = {
      jobId: 'qj_long_running',
      jobType: 'check_qlib_data',
      label: '检查 Qlib 数据',
      status: 'running',
      params: {},
      createdAt: '2026-07-01T09:00:00+08:00',
      updatedAt: '2026-07-01T09:00:00+08:00',
      artifactPaths: [],
    };
    const completedHistory = Array.from({ length: 30 }, (_, index) => ({
      jobId: `qj_history_${index}`,
      jobType: 'report_data_quality',
      label: '生成数据质量报告',
      status: 'completed',
      params: {},
      createdAt: `2026-07-${String(index + 2).padStart(2, '0')}T10:00:00+08:00`,
      updatedAt: `2026-07-${String(index + 2).padStart(2, '0')}T10:01:00+08:00`,
      artifactPaths: [],
    }));
    getQuantStatus.mockResolvedValueOnce({
      availableJobTypes: {},
      activeCount: 1,
      latestJobs: { check_qlib_data: longRunningJob },
      recentJobs: [longRunningJob],
    });
    listQuantJobs.mockResolvedValueOnce({
      total: completedHistory.length,
      items: completedHistory,
    });

    renderWorkflowPage();

    expect(await screen.findByText('qj_history_29')).toBeInTheDocument();
    expect(screen.getByText('1 个运行中')).toBeInTheDocument();
    expect(screen.getByText('qj_long_running')).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: '取消 检查 Qlib 数据' }),
    ).toBeInTheDocument();
  });

  it('removes a stale active job when the authoritative status snapshot no longer lists it', async () => {
    const staleRunningJob = {
      jobId: 'qj_status_stale_running',
      jobType: 'check_qlib_data',
      label: '检查 Qlib 数据',
      status: 'running',
      params: {},
      createdAt: '2026-07-01T09:00:00+08:00',
      updatedAt: '2026-07-01T09:00:00+08:00',
      artifactPaths: [],
    };
    getQuantStatus
      .mockResolvedValueOnce({
        availableJobTypes: {},
        activeCount: 1,
        latestJobs: { check_qlib_data: staleRunningJob },
        recentJobs: [staleRunningJob],
      })
      .mockResolvedValueOnce({
        availableJobTypes: {},
        activeCount: 0,
        latestJobs: {},
        recentJobs: [],
      });
    listQuantJobs
      .mockResolvedValueOnce({ total: 1, items: [staleRunningJob] })
      .mockResolvedValueOnce({ total: 0, items: [] });

    renderWorkflowPage();
    expect(await screen.findByText('qj_status_stale_running')).toBeInTheDocument();
    expect(screen.getByText('1 个运行中')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: '刷新' }));
    await waitFor(() => {
      expect(getQuantStatus).toHaveBeenCalledTimes(2);
    });

    expect(await screen.findByText('0 个运行中')).toBeInTheDocument();
    expect(screen.queryByText('qj_status_stale_running')).not.toBeInTheDocument();
    expect(
      screen.queryByRole('button', { name: '取消 检查 Qlib 数据' }),
    ).not.toBeInTheDocument();
  });

  it('does not resurrect a stale active job from a delayed history response', async () => {
    const staleRunningJob = {
      jobId: 'qj_delayed_history_stale',
      jobType: 'check_qlib_data',
      label: '检查 Qlib 数据',
      status: 'running',
      params: {},
      createdAt: '2026-07-01T09:00:00+08:00',
      updatedAt: '2026-07-01T09:00:00+08:00',
      artifactPaths: [],
    };
    getQuantStatus.mockResolvedValueOnce({
      availableJobTypes: {},
      activeCount: 0,
      latestJobs: {},
      recentJobs: [],
    });
    let resolveList: (value: unknown) => void = () => undefined;
    listQuantJobs.mockReturnValueOnce(new Promise((resolve) => {
      resolveList = resolve;
    }));

    renderWorkflowPage();
    await waitFor(() => {
      expect(getQuantStatus).toHaveBeenCalledTimes(1);
    });
    await act(async () => {
      resolveList({ total: 1, items: [staleRunningJob] });
    });

    expect(await screen.findByText('0 个运行中')).toBeInTheDocument();
    expect(screen.queryByText('qj_delayed_history_stale')).not.toBeInTheDocument();
  });

  it('removes a stale active job when the authoritative stream snapshot no longer lists it', async () => {
    const staleRunningJob = {
      jobId: 'qj_stream_stale_running',
      jobType: 'check_qlib_data',
      label: '检查 Qlib 数据',
      status: 'running',
      params: {},
      createdAt: '2026-07-01T09:00:00+08:00',
      updatedAt: '2026-07-01T09:00:00+08:00',
      artifactPaths: [],
    };
    getQuantStatus.mockResolvedValueOnce({
      availableJobTypes: {},
      activeCount: 1,
      latestJobs: { check_qlib_data: staleRunningJob },
      recentJobs: [staleRunningJob],
    });
    listQuantJobs.mockResolvedValueOnce({ total: 1, items: [staleRunningJob] });

    renderWorkflowPage();
    expect(await screen.findByText('qj_stream_stale_running')).toBeInTheDocument();
    expect(screen.getByText('1 个运行中')).toBeInTheDocument();

    act(() => {
      FakeEventSource.instances[0].emit('jobs', { jobs: [] });
    });

    expect(await screen.findByText('0 个运行中')).toBeInTheDocument();
    expect(screen.queryByText('qj_stream_stale_running')).not.toBeInTheDocument();
    expect(
      screen.queryByRole('button', { name: '取消 检查 Qlib 数据' }),
    ).not.toBeInTheDocument();
  });

  it('updates the active job count immediately after cancelling a running job', async () => {
    const runningJob = {
      jobId: 'qj_running',
      jobType: 'check_qlib_data',
      label: '检查 Qlib 数据',
      status: 'running',
      params: {},
      command: [],
      cwd: '/Users/cq/WorkingProjects',
      createdAt: '2026-07-03T22:00:00+08:00',
      updatedAt: '2026-07-03T22:00:00+08:00',
      logPath: '/tmp/job.log',
      artifactPaths: [],
    };
    getQuantStatus.mockResolvedValueOnce({
      availableJobTypes: {
        check_qlib_data: { label: '检查 Qlib 数据', allowedParams: [] },
      },
      activeCount: 1,
      latestJobs: {
        check_qlib_data: runningJob,
      },
      recentJobs: [runningJob],
    });
    listQuantJobs.mockResolvedValueOnce({
      total: 1,
      items: [runningJob],
    });
    cancelQuantJob.mockResolvedValue({
      ...runningJob,
      status: 'cancelled',
      completedAt: '2026-07-03T22:01:00+08:00',
      updatedAt: '2026-07-03T22:01:00+08:00',
      error: 'Cancelled by user from the web UI.',
    });

    renderWorkflowPage();

    expect(await screen.findByText('1 个运行中')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: '取消 检查 Qlib 数据' }));
    expect(cancelQuantJob).not.toHaveBeenCalled();
    expect(screen.getByRole('dialog', { name: '确认取消任务' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: '确认取消' }));

    await waitFor(() => {
      expect(cancelQuantJob).toHaveBeenCalledWith('qj_running');
    });
    expect(await screen.findByText('0 个运行中')).toBeInTheDocument();
  });

  it('updates the job list from the quant job stream', async () => {
    renderWorkflowPage();

    await screen.findByText('实时检查通过');
    expect(FakeEventSource.instances[0].url).toBe('/api/v1/quant/jobs/stream');
    expect(FakeEventSource.instances[0].init).toEqual({ withCredentials: true });

    act(() => {
      FakeEventSource.instances[0].emit('jobs', {
        jobs: [
          {
            jobId: 'qj_stream',
            jobType: 'check_qlib_data',
            label: '检查 Qlib 数据',
            status: 'completed',
            params: {},
            command: [],
            cwd: '/Users/cq/WorkingProjects',
            createdAt: '2026-07-04T01:00:00+08:00',
            updatedAt: '2026-07-04T01:00:01+08:00',
            logPath: '/tmp/job.log',
            artifactPaths: ['/tmp/qlib_check.log'],
          },
        ],
      });
    });

    expect(await screen.findByText('qj_stream')).toBeInTheDocument();
    expect(screen.getByText('qlib_check.log')).toBeInTheDocument();
  });

  it('refreshes an open log after streamed updates and identifies truncated output', async () => {
    const runningJob = {
      jobId: 'qj_live_log',
      jobType: 'check_qlib_data',
      label: '检查 Qlib 数据',
      status: 'running',
      params: {},
      createdAt: '2026-08-02T10:00:00+08:00',
      updatedAt: '2026-08-02T10:00:00+08:00',
      artifactPaths: [],
    };
    getQuantStatus.mockResolvedValueOnce({
      availableJobTypes: {},
      activeCount: 1,
      latestJobs: { check_qlib_data: runningJob },
      recentJobs: [runningJob],
    });
    listQuantJobs.mockResolvedValueOnce({ total: 1, items: [runningJob] });
    getQuantJobLog
      .mockResolvedValueOnce({ jobId: runningJob.jobId, content: 'stage 1', truncated: false })
      .mockResolvedValueOnce({ jobId: runningJob.jobId, content: 'stage 2 tail', truncated: true });

    renderWorkflowPage();
    fireEvent.click(await screen.findByRole('button', { name: '日志' }));
    expect(await screen.findByText('stage 1')).toBeInTheDocument();

    act(() => {
      FakeEventSource.instances[0].emit('jobs', {
        jobs: [{ ...runningJob, updatedAt: '2026-08-02T10:00:02+08:00' }],
      });
    });

    expect(await screen.findByText('stage 2 tail')).toBeInTheDocument();
    expect(screen.getByText('日志过长，仅显示最后一部分。')).toBeInTheDocument();
  });

  it('keeps the newest job per type when stream events arrive unsorted', async () => {
    renderWorkflowPage();

    await screen.findByText('实时检查通过');

    act(() => {
      FakeEventSource.instances[0].emit('jobs', {
        jobs: [
          {
            jobId: 'qj_old_check',
            jobType: 'check_qlib_data',
            label: '检查 Qlib 数据',
            status: 'completed',
            params: {},
            command: [],
            cwd: '/Users/cq/WorkingProjects',
            createdAt: '2026-07-04T01:00:00+08:00',
            updatedAt: '2026-07-04T01:00:01+08:00',
            logPath: '/tmp/job.log',
            artifactPaths: ['/tmp/old_report.md'],
          },
          {
            jobId: 'qj_new_check',
            jobType: 'check_qlib_data',
            label: '检查 Qlib 数据',
            status: 'completed',
            params: {},
            command: [],
            cwd: '/Users/cq/WorkingProjects',
            createdAt: '2026-07-04T02:00:00+08:00',
            updatedAt: '2026-07-04T02:00:01+08:00',
            logPath: '/tmp/job.log',
            artifactPaths: ['/tmp/new_report.md'],
          },
        ],
      });
    });

    expect(await screen.findByRole('link', { name: 'new_report.md' })).toBeInTheDocument();
    expect(screen.queryByRole('link', { name: 'old_report.md' })).not.toBeInTheDocument();
  });

  it('does not let a stale list response replace a newer status job', async () => {
    const newer = {
      jobId: 'qj_newer',
      jobType: 'check_qlib_data',
      label: '检查 Qlib 数据',
      status: 'running',
      params: {},
      createdAt: '2026-07-04T02:00:00.200000+08:00',
      updatedAt: '2026-07-04T02:00:00.300000+08:00',
      artifactPaths: ['/tmp/newer.md'],
    };
    let resolveList: (value: unknown) => void = () => undefined;
    getQuantStatus.mockResolvedValueOnce({
      availableJobTypes: {},
      activeCount: 1,
      latestJobs: { check_qlib_data: newer },
      recentJobs: [newer],
    });
    listQuantJobs.mockReturnValueOnce(new Promise((resolve) => {
      resolveList = resolve;
    }));

    renderWorkflowPage();
    expect(await screen.findByRole('link', { name: 'newer.md' })).toBeInTheDocument();

    await act(async () => {
      resolveList({
        total: 1,
        items: [{
          ...newer,
          jobId: 'qj_older',
          status: 'completed',
          createdAt: '2026-07-04T01:00:00.100000+08:00',
          updatedAt: '2026-07-04T03:00:00.100000+08:00',
          artifactPaths: ['/tmp/older.md'],
        }],
      });
    });

    expect(screen.getByRole('link', { name: 'newer.md' })).toBeInTheDocument();
    expect(screen.queryByRole('link', { name: 'older.md' })).not.toBeInTheDocument();
  });

  it('reconnects the quant job stream and localizes the transient error', async () => {
    localStorage.setItem(UI_LANGUAGE_STORAGE_KEY, 'en');
    render(
      <UiLanguageProvider>
        <MemoryRouter>
          <QuantTaskPage surface="all" />
        </MemoryRouter>
      </UiLanguageProvider>,
    );

    await screen.findByText('Live checks passed');
    expect(FakeEventSource.instances).toHaveLength(1);

    vi.useFakeTimers();

    act(() => {
      FakeEventSource.instances[0].fail();
    });

    expect(screen.getByText('Live job connection was interrupted. Reconnecting...')).toBeInTheDocument();
    expect(FakeEventSource.instances[0].closed).toBe(true);

    act(() => {
      vi.advanceTimersByTime(999);
    });
    expect(FakeEventSource.instances).toHaveLength(1);

    act(() => {
      vi.advanceTimersByTime(1);
    });
    expect(FakeEventSource.instances).toHaveLength(2);

    act(() => {
      FakeEventSource.instances[1].open();
    });

    expect(screen.queryByText('Live job connection was interrupted. Reconnecting...')).not.toBeInTheDocument();
  });

  it('reconnects when the server explicitly ends a bounded job stream', async () => {
    renderWorkflowPage();

    await screen.findByText('实时检查通过');
    vi.useFakeTimers();

    act(() => {
      FakeEventSource.instances[0].emit('stream_end', { type: 'stream_end' });
    });

    expect(FakeEventSource.instances[0].closed).toBe(true);
    expect(screen.queryByText('任务实时连接已断开，正在尝试重连。')).not.toBeInTheDocument();

    act(() => {
      vi.advanceTimersByTime(0);
    });

    expect(FakeEventSource.instances).toHaveLength(2);
  });

  it('does not schedule a second reconnect when stream end is followed by an error', async () => {
    renderWorkflowPage();

    await screen.findByText('实时检查通过');
    vi.useFakeTimers();

    act(() => {
      FakeEventSource.instances[0].emit('stream_end', { type: 'stream_end' });
      FakeEventSource.instances[0].fail();
      vi.advanceTimersByTime(0);
    });
    expect(FakeEventSource.instances).toHaveLength(2);

    act(() => {
      vi.advanceTimersByTime(1000);
    });
    expect(FakeEventSource.instances).toHaveLength(2);
  });

  it('backs off repeated stream reconnect attempts until the connection opens', async () => {
    renderWorkflowPage();

    await screen.findByText('实时检查通过');
    vi.useFakeTimers();

    act(() => {
      FakeEventSource.instances[0].fail();
      vi.advanceTimersByTime(1000);
    });
    expect(FakeEventSource.instances).toHaveLength(2);

    act(() => {
      FakeEventSource.instances[1].fail();
      vi.advanceTimersByTime(1999);
    });
    expect(FakeEventSource.instances).toHaveLength(2);

    act(() => {
      vi.advanceTimersByTime(1);
    });
    expect(FakeEventSource.instances).toHaveLength(3);

    act(() => {
      FakeEventSource.instances[2].open();
      FakeEventSource.instances[2].fail();
      vi.advanceTimersByTime(1000);
    });
    expect(FakeEventSource.instances).toHaveLength(4);
  });
});
