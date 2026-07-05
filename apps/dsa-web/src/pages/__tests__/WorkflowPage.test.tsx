import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { createParsedApiError } from '../../api/error';
import { UiLanguageProvider } from '../../contexts/UiLanguageContext';
import { UI_LANGUAGE_STORAGE_KEY } from '../../utils/uiLanguage';
import WorkflowPage from '../WorkflowPage';

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

const { getQuantStatus, listQuantJobs, submitQuantJob, getQuantJobLog, getQuantJobArtifactText, cancelQuantJob } = vi.hoisted(() => ({
  getQuantStatus: vi.fn(),
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
    getStatus: () => Promise.resolve({
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
    }),
  },
}));

function renderWorkflowPage() {
  localStorage.setItem(UI_LANGUAGE_STORAGE_KEY, 'zh');
  return render(
    <UiLanguageProvider>
      <MemoryRouter>
        <WorkflowPage />
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

    getQuantStatus.mockResolvedValue({
      availableJobTypes: {
        check_qlib_data: { label: '检查 Qlib 数据', allowedParams: [] },
        report_market_regime: { label: '生成市场状态和现金仓位信号', allowedParams: ['lookback_days', 'min_symbols', 'limit'] },
        report_cash_control_backtest: { label: '生成现金仓位叠加回测', allowedParams: ['lookback_days', 'min_symbols', 'limit'] },
        report_qlib_predictions: { label: '生成 Qlib 预测摘要', allowedParams: ['run_id', 'topk'] },
        train_baseline_lgb: { label: '训练 Qlib LightGBM 基线', allowedParams: ['config'] },
        run_china_astock_comparison: { label: '生成 china-astock 策略对比', allowedParams: [] },
        optimize_strategy: { label: '策略优化预演', allowedParams: ['dry_run', 'phase', 'topk'] },
      },
      activeCount: 0,
      latestJobs: {
        train_baseline_lgb: {
          jobId: 'qj_train',
          jobType: 'train_baseline_lgb',
          label: '训练 Qlib LightGBM 基线',
          status: 'completed',
          params: {},
          command: [],
          cwd: '/Users/cq/WorkingProjects',
          createdAt: '2026-07-03T21:00:00+08:00',
          updatedAt: '2026-07-03T21:10:00+08:00',
          logPath: '/tmp/job.log',
          artifactPaths: ['/Users/cq/WorkingProjects/quant_workflow/reports/latest_training_result.json'],
        },
      },
      recentJobs: [],
    });
    listQuantJobs.mockResolvedValue({
      total: 1,
      items: [
        {
          jobId: 'qj_train',
          jobType: 'train_baseline_lgb',
          label: '训练 Qlib LightGBM 基线',
          status: 'completed',
          params: {},
          command: [],
          cwd: '/Users/cq/WorkingProjects',
          createdAt: '2026-07-03T21:00:00+08:00',
          updatedAt: '2026-07-03T21:10:00+08:00',
          logPath: '/tmp/job.log',
          artifactPaths: ['/Users/cq/WorkingProjects/quant_workflow/reports/latest_training_result.json'],
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

  it('renders a Chinese web-first quant workflow console', async () => {
    renderWorkflowPage();

    expect(screen.getByRole('heading', { name: '量化任务中心' })).toBeInTheDocument();
    expect(await screen.findByText('当前数据可训练')).toBeInTheDocument();
    expect(screen.getByText('MLflow 原始记录')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: '打开 MLflow，新窗口' })).toHaveAttribute('href', 'http://127.0.0.1:6006/');
    expect(screen.getByText('2020-01-02 ~ 2026-07-03')).toBeInTheDocument();
    expect(screen.getAllByText('更新数据')).toHaveLength(2);
    expect(screen.getByText('合并 Qlib')).toBeInTheDocument();
    expect(screen.getByText('应用清洗策略')).toBeInTheDocument();
    expect(screen.getAllByText('质量报告').length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('训练+回测')).toBeInTheDocument();
    expect(screen.getByText('生成市场状态和现金仓位信号')).toBeInTheDocument();
    expect(screen.getByLabelText('生成市场状态和现金仓位信号 回看交易日')).toBeInTheDocument();
    expect(screen.getByText('生成现金仓位叠加回测')).toBeInTheDocument();
    expect(screen.getByLabelText('生成现金仓位叠加回测 回看交易日')).toBeInTheDocument();
    expect(screen.getByText('生成 Qlib 预测摘要')).toBeInTheDocument();
    expect(screen.getByLabelText('生成 Qlib 预测摘要 TopK')).toBeInTheDocument();
    expect(screen.getByText('生成 china-astock 策略对比')).toBeInTheDocument();
    expect(screen.getByText('读取 china-astock-quant 既有搜索和赢家评估结果，生成策略对比 CSV/Markdown 报告；只读报告，不连接券商或交易接口。')).toBeInTheDocument();
    expect(screen.getByText('今日流程')).toBeInTheDocument();
    expect(screen.getByText('按顺序完成检查、报告、预演、训练和记录。')).toBeInTheDocument();
    expect(screen.getByText('质量和异常')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '今日流程 清洗预演 预演' })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: '实验记录' })).toHaveAttribute('href', '/research');
    expect(screen.getByRole('link', { name: '回测页' })).toHaveAttribute('href', '/backtest');
    expect(screen.getByText('检查 Qlib 数据')).toBeInTheDocument();
    expect(screen.getAllByText('训练 Qlib LightGBM 基线')).toHaveLength(2);
    expect(screen.getByText('latest_training_result.json')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'latest_training_result.json' })).toHaveAttribute(
      'href',
      '/api/v1/quant/jobs/qj_train/artifacts/0',
    );
  });

  it('submits a whitelisted quant job from the workflow page', async () => {
    renderWorkflowPage();

    fireEvent.click(await screen.findByRole('button', { name: '运行 检查 Qlib 数据' }));

    await waitFor(() => {
      expect(submitQuantJob).toHaveBeenCalledWith('check_qlib_data', {});
    });
    expect(await screen.findByText('qj_check')).toBeInTheDocument();
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

  it('submits the selected Qlib provider for training jobs', async () => {
    renderWorkflowPage();

    fireEvent.change(await screen.findByLabelText('训练 Qlib LightGBM 基线 数据版本'), {
      target: { value: 'cleaned' },
    });
    fireEvent.click(screen.getByRole('button', { name: '运行 训练 Qlib LightGBM 基线' }));

    await waitFor(() => {
      expect(submitQuantJob).toHaveBeenCalledWith('train_baseline_lgb', {
        provider: 'cleaned',
      });
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

  it('submits browser-safe strategy optimization dry-run parameters', async () => {
    renderWorkflowPage();

    fireEvent.change(await screen.findByLabelText('策略优化预演 阶段'), {
      target: { value: '4' },
    });
    fireEvent.change(screen.getByLabelText('策略优化预演 TopK'), {
      target: { value: '30' },
    });
    fireEvent.click(screen.getByRole('button', { name: '运行 策略优化预演' }));

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
          <WorkflowPage />
        </MemoryRouter>
      </UiLanguageProvider>,
    );

    await screen.findByText('当前数据可训练');
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
