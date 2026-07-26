import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { createParsedApiError } from '../../api/error';
import { UiLanguageProvider } from '../../contexts/UiLanguageContext';
import { UI_LANGUAGE_STORAGE_KEY } from '../../utils/uiLanguage';
import DataPage from '../DataPage';
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

const { getQuantStatus, getResearchStatus, listQuantJobs, submitQuantJob, getQuantJobLog, getQuantJobArtifactText, cancelQuantJob } = vi.hoisted(() => ({
  getQuantStatus: vi.fn(),
  getResearchStatus: vi.fn(),
  listQuantJobs: vi.fn(),
  submitQuantJob: vi.fn(),
  getQuantJobLog: vi.fn(),
  getQuantJobArtifactText: vi.fn(),
  cancelQuantJob: vi.fn(),
}));

vi.mock('../../api/quant', () => ({
  quantApi: {
    getStatus: () => getQuantStatus(),
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
      dataSources: [
        { name: 'baostock', files: 5300, rows: 3000000, dateRange: '2020-01-02 ~ 2026-07-03', status: 'available' },
      ],
      qlibProvider: {
        path: '/Users/cq/WorkingProjects/quant_data/qlib/cn_data_baostock_jqdata',
        calendarDays: 1500,
        calendarRange: '2020-01-02 ~ 2026-07-03',
        instruments: 5300,
      },
      latestTraining: {
        status: 'completed',
        experimentName: 'baseline_lgb_alpha158',
        runId: 'abc123',
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

function renderWorkflowPage() {
  localStorage.setItem(UI_LANGUAGE_STORAGE_KEY, 'zh');
  return render(
    <UiLanguageProvider>
      <MemoryRouter>
        <QuantTaskPage surface="all" />
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

    getQuantStatus.mockResolvedValue({
      availableJobTypes: {
        prepare_baostock_refresh: { label: '准备 Baostock 增量清单', allowedParams: ['target_end', 'limit'] },
        download_baostock: { label: '下载 Baostock 长历史', allowedParams: ['start', 'end', 'sleep', 'limit', 'retries', 'timeout', 'max_consecutive_failures', 'max_requests', 'overlap_days', 'retry_failed', 'retry_empty', 'retry_timeout', 'incremental'] },
        download_eastmoney: { label: '下载东方财富日线', allowedParams: ['start', 'end', 'sleep', 'limit', 'retries', 'timeout', 'max_consecutive_failures', 'max_requests', 'retry_failed', 'incremental', 'overlap_days', 'use_system_proxy'] },
        download_akshare: { label: '下载 AkShare 日线', allowedParams: ['start', 'end', 'sleep', 'limit', 'retries', 'timeout', 'max_consecutive_failures', 'max_requests', 'retry_failed', 'incremental', 'overlap_days', 'use_system_proxy'] },
        validate_latest_snapshot: { label: '校验新浪最新快照', allowedParams: ['limit', 'timeout', 'use_system_proxy'] },
        download_jqdata: { label: '验证 JQData 连接（dry-run）', allowedParams: ['universe', 'start', 'end', 'limit', 'sleep', 'dry_run'] },
        merge_to_qlib: { label: '合并并转换为 Qlib', allowedParams: ['no_overwrite', 'smoke'] },
        apply_data_cleaning: { label: '应用清洗策略生成 cleaned Qlib', allowedParams: ['dry_run', 'no_overwrite', 'skip_qlib_convert'] },
        repair_qlib_provider: { label: '预演修复 Qlib Provider', allowedParams: [] },
        check_qlib_data: { label: '检查 Qlib 数据', allowedParams: [] },
        report_data_quality: { label: '生成数据质量报告', allowedParams: ['overlap_sample', 'merged_sample'] },
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
        run_candidate_pipeline: { label: '运行五个选股模型候选流程', allowedParams: ['dry_run', 'provider'] },
        generate_csi300: { label: '预演生成 CSI300 股票池', allowedParams: [] },
        add_index_benchmark: { label: '预演添加沪深300 benchmark', allowedParams: [] },
        run_china_astock_comparison: { label: '生成 china-astock 策略对比', allowedParams: [] },
        report_china_astock_indicators: { label: '生成 china-astock 技术指标报告', allowedParams: ['symbols', 'lookback_days', 'limit'] },
        optimize_strategy: { label: '交易策略优化预演', allowedParams: ['dry_run', 'phase', 'topk'] },
        run_dsa_backtest: { label: '运行 DSA 分析准确性回测', allowedParams: ['code', 'eval_window_days', 'force'] },
      },
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
      '/backtest?tab=trained&runId=abc123&modelId=latest&strategyId=baseline_daily',
    );
  });

  it('blocks formal training when the data gate fails but still allows the explicit dry-run preview', async () => {
    getResearchStatus.mockResolvedValue({
      ...readyResearchStatus(),
      trainable: false,
      trainingReadiness: {
        status: 'blocked_data',
        summary: 'SH000300 benchmark is missing or incomplete (8710 missing days)',
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

  it('separates data tasks from the remaining quant tasks', async () => {
    localStorage.setItem(UI_LANGUAGE_STORAGE_KEY, 'zh');
    const dataView = render(
      <UiLanguageProvider>
        <MemoryRouter initialEntries={['/data']}>
          <DataPage />
        </MemoryRouter>
      </UiLanguageProvider>,
    );

    expect(await screen.findByRole('heading', { name: '数据更新与检查' })).toBeInTheDocument();
    expect(screen.getByTestId('workflow-job-run_daily_update')).toBeInTheDocument();
    expect(screen.getByTestId('workflow-job-check_qlib_data')).toBeInTheDocument();
    expect(screen.queryByTestId('workflow-job-train_baseline_lgb')).not.toBeInTheDocument();
    expect(screen.queryByTestId('workflow-job-report_qlib_predictions')).not.toBeInTheDocument();

    dataView.unmount();
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
    expect(screen.queryByTestId('workflow-job-run_daily_update')).not.toBeInTheDocument();
    expect(screen.queryByTestId('workflow-job-train_baseline_lgb')).not.toBeInTheDocument();
  });

  it('renders a Chinese web-first quant workflow console', async () => {
    renderWorkflowPage();

    expect(screen.getByRole('heading', { name: '量化任务中心' })).toBeInTheDocument();
    expect(await screen.findByText('当前数据可训练')).toBeInTheDocument();
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
    expect(screen.getByText('合并 Qlib')).toBeInTheDocument();
    expect(screen.getByText('校验新浪最新快照')).toBeInTheDocument();
    expect(screen.getByText('抽样读取新浪行情快照，对比合并后的最新收盘价；只生成校验报告，不参与训练。')).toBeInTheDocument();
    expect(screen.getByText('预演生成 CSI300 股票池')).toBeInTheDocument();
    expect(screen.getByText('预演添加沪深300 benchmark')).toBeInTheDocument();
    expect(screen.queryByLabelText('修复 Qlib Provider 只检查不写入')).not.toBeInTheDocument();
    expect(screen.getByText('应用清洗策略')).toBeInTheDocument();
    expect(screen.getAllByText('质量报告').length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('选股模型训练与交易策略验证')).toBeInTheDocument();
    expect(screen.getByText('运行五个选股模型候选流程')).toBeInTheDocument();
    expect(screen.getByTestId('workflow-job-run_candidate_pipeline')).toBeInTheDocument();
    expect(screen.getByTestId('workflow-run-run_candidate_pipeline')).toHaveAccessibleName('运行 运行五个选股模型候选流程');
    expect(screen.getByLabelText('运行五个选股模型候选流程 数据版本')).toBeInTheDocument();
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
    expect(screen.getByText('复用 china-astock-quant 的 SMA/EMA/RSI/ATR/Bollinger 指标函数，读取本地 cleaned 行情生成只读技术面观察报告，不连接券商、不自动下单。')).toBeInTheDocument();
    expect(screen.getByLabelText('生成 china-astock 技术指标报告 股票代码')).toBeInTheDocument();
    expect(screen.getByText('生成数据治理盘点和清理预案')).toBeInTheDocument();
    expect(screen.getByTestId('workflow-job-report_data_hygiene')).toBeInTheDocument();
    expect(screen.getByTestId('workflow-run-report_data_hygiene')).toHaveAccessibleName('运行 生成数据治理盘点和清理预案');
    expect(screen.getByText('只读盘点当前数据、日志、旧 provider 和临时脚本，生成 inventory 与 dry-run 清理计划；不移动、不删除、不改配置。')).toBeInTheDocument();
    expect(screen.getByText('生成工作台能力盘点')).toBeInTheDocument();
    expect(screen.getByText('只读扫描 DSA 页面/API、quant_workflow 脚本、Qlib 接入和 china-astock-quant 可复用能力，输出功能入口和缺口清单。')).toBeInTheDocument();
    expect(screen.getByText('今日流程')).toBeInTheDocument();
    expect(screen.getByText('按顺序完成检查、报告、预演、训练和记录。')).toBeInTheDocument();
    expect(screen.getByText('质量和异常')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '今日流程 清洗预演 预演' })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: '实验记录' })).toHaveAttribute('href', '/research');
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

  it('disables workflow jobs that are not advertised by the backend whitelist', async () => {
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
    getQuantStatus.mockReturnValueOnce(new Promise((resolve) => {
      resolveStatus = resolve;
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
        activeCount: 0,
        latestJobs: {},
        recentJobs: [],
      });
    });

    await waitFor(() => {
      expect(hygieneButton).not.toBeDisabled();
    });
    expect(within(hygieneHeadingCluster as HTMLElement).queryByText('后端未启用')).not.toBeInTheDocument();
  });

  it('keeps daily guide job shortcuts disabled until the backend whitelist loads', async () => {
    let resolveStatus: (value: unknown) => void = () => undefined;
    getQuantStatus.mockReturnValueOnce(new Promise((resolve) => {
      resolveStatus = resolve;
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
          apply_data_cleaning: { label: '应用清洗策略生成 cleaned Qlib', allowedParams: ['dry_run', 'no_overwrite'] },
        },
        activeCount: 0,
        latestJobs: {},
        recentJobs: [],
      });
    });

    await waitFor(() => {
      expect(guideButton).not.toBeDisabled();
    });
  });

  it('disables daily guide job shortcuts that are missing from the backend whitelist', async () => {
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

    fireEvent.click(await screen.findByRole('button', { name: '运行 预演生成 CSI300 股票池' }));
    fireEvent.click(await screen.findByRole('button', { name: '运行 预演添加沪深300 benchmark' }));

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

    fireEvent.click(await screen.findByLabelText('合并并转换为 Qlib 使用隔离小样本演练'));
    fireEvent.click(screen.getByLabelText('合并并转换为 Qlib 跳过已有输出文件'));
    fireEvent.click(screen.getByRole('button', { name: '运行 合并并转换为 Qlib' }));

    await waitFor(() => {
      expect(submitQuantJob).toHaveBeenCalledWith('merge_to_qlib', {
        smoke: true,
        no_overwrite: true,
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

  it('submits sampled data quality report parameters from the workflow page', async () => {
    renderWorkflowPage();

    fireEvent.change(await screen.findByLabelText('生成数据质量报告 重叠样本数'), {
      target: { value: '5' },
    });
    fireEvent.change(screen.getByLabelText('生成数据质量报告 合并样本数'), {
      target: { value: '5' },
    });
    fireEvent.click(screen.getByRole('button', { name: '运行 生成数据质量报告' }));

    await waitFor(() => {
      expect(submitQuantJob).toHaveBeenCalledWith('report_data_quality', {
        overlap_sample: '5',
        merged_sample: '5',
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

    fireEvent.click(await screen.findByLabelText('应用清洗策略 只预演不写入'));
    fireEvent.click(await screen.findByLabelText('应用清洗策略 不覆盖已有文件'));
    fireEvent.click(screen.getByRole('button', { name: '运行 应用清洗策略' }));

    await waitFor(() => {
      expect(submitQuantJob).toHaveBeenCalledWith('apply_data_cleaning', {
        dry_run: true,
        no_overwrite: true,
      });
    });
  });

  it('runs the daily guide cleaning preflight with safe dry-run parameters', async () => {
    renderWorkflowPage();

    fireEvent.click(await screen.findByRole('button', { name: '今日流程 清洗预演 预演' }));

    await waitFor(() => {
      expect(submitQuantJob).toHaveBeenCalledWith('apply_data_cleaning', {
        dry_run: true,
        no_overwrite: true,
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

    await waitFor(() => {
      expect(cancelQuantJob).toHaveBeenCalledWith('qj_running');
    });
    expect(await screen.findByText('0 个运行中')).toBeInTheDocument();
  });

  it('updates the job list from the quant job stream', async () => {
    renderWorkflowPage();

    await screen.findByText('当前数据可训练');
    expect(FakeEventSource.instances[0].url).toBe('/api/v1/quant/jobs/stream');
    expect(FakeEventSource.instances[0].init).toEqual({ withCredentials: true });

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

    expect(await screen.findByText('qj_stream')).toBeInTheDocument();
    expect(screen.getByText('qlib_check.log')).toBeInTheDocument();
  });

  it('keeps the newest job per type when stream events arrive unsorted', async () => {
    renderWorkflowPage();

    await screen.findByText('当前数据可训练');

    FakeEventSource.instances[0].emit('jobs', {
      jobs: [
        {
          jobId: 'qj_old_quality',
          jobType: 'report_data_quality',
          label: '生成数据质量报告',
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
          jobId: 'qj_new_quality',
          jobType: 'report_data_quality',
          label: '生成数据质量报告',
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

    expect(await screen.findByRole('link', { name: 'new_report.md' })).toBeInTheDocument();
    expect(screen.queryByRole('link', { name: 'old_report.md' })).not.toBeInTheDocument();
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

    await screen.findByText('Data ready for training');
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

    await screen.findByText('当前数据可训练');
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

  it('backs off repeated stream reconnect attempts until the connection opens', async () => {
    renderWorkflowPage();

    await screen.findByText('当前数据可训练');
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
