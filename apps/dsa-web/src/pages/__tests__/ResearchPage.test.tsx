import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { UiLanguageProvider } from '../../contexts/UiLanguageContext';
import { UI_LANGUAGE_STORAGE_KEY } from '../../utils/uiLanguage';
import ResearchPage from '../ResearchPage';

const { getResearchStatus, archiveExperimentRun, permanentlyDeleteExperimentRun } = vi.hoisted(() => ({
  getResearchStatus: vi.fn(),
  archiveExperimentRun: vi.fn(),
  permanentlyDeleteExperimentRun: vi.fn(),
}));

vi.mock('../../api/research', () => ({
  researchApi: {
    getStatus: () => getResearchStatus(),
    archiveExperimentRun: (runId: string) => archiveExperimentRun(runId),
    permanentlyDeleteExperimentRun: (runId: string) => permanentlyDeleteExperimentRun(runId),
  },
}));

function renderResearchPage(language: 'zh' | 'en' = 'zh') {
  localStorage.setItem(UI_LANGUAGE_STORAGE_KEY, language);

  return render(
    <UiLanguageProvider>
      <MemoryRouter>
        <ResearchPage />
      </MemoryRouter>
    </UiLanguageProvider>,
  );
}

describe('ResearchPage', () => {
  beforeEach(() => {
    localStorage.clear();
    getResearchStatus.mockReset();
    archiveExperimentRun.mockReset();
    permanentlyDeleteExperimentRun.mockReset();
    getResearchStatus.mockResolvedValue({
      generatedAt: '2026-07-02T10:00:00+08:00',
      trainable: true,
      trainingReadiness: {
        status: 'trainable',
        summary: 'Qlib provider has enough CSI300 coverage.',
      },
      dataSources: [
        {
          name: 'akshare',
          files: 899,
          rows: 1000,
          dateRange: '2005-01-04 ~ 2026-06-10',
          status: 'partial',
        },
        {
          name: 'jqdata',
          files: 5157,
          rows: 343237,
          dateRange: '2025-03-03 ~ 2025-06-10',
          status: 'available',
        },
      ],
      qlibProvider: {
        path: '/Users/cq/WorkingProjects/quant_data/qlib/cn_data_baostock_jqdata',
        calendarDays: 1316,
        calendarRange: '2020-01-02 ~ 2025-06-11',
        instruments: 5340,
      },
      latestTraining: {
        status: 'completed',
        experimentName: 'baseline_lgb_alpha158_cleaned',
        runId: 'cleaned123',
        provider: 'cleaned',
        providerUri: '/Users/cq/WorkingProjects/quant_data/qlib/cn_data_baostock_jqdata_cleaned',
        metrics: {
          '1day.excess_return_with_cost.annualized_return': 0.198637,
          '1day.excess_return_with_cost.information_ratio': 1.705515,
          '1day.excess_return_with_cost.max_drawdown': -0.091497,
        },
      },
      experimentNotebook: {
        status: 'available',
        totalRuns: 2,
        latestNote: '# 实验记录本\n\n## 2026-07-03 baseline_lgb_alpha158\n\n结论：长历史 baseline 已跑通，但收益质量偏弱。\n\n下一步：补充风险控制和市场择时实验。',
        logPath: '/Users/cq/WorkingProjects/quant_workflow/reports/experiment_log.md',
        runsPath: '/Users/cq/WorkingProjects/quant_workflow/reports/experiment_runs.jsonl',
        runs: [
          {
            runId: 'cleaned123',
            experimentName: 'baseline_lgb_alpha158_cleaned',
            status: 'completed',
            provider: 'cleaned',
            qlibProvider: '/Users/cq/WorkingProjects/quant_data/qlib/cn_data_baostock_jqdata_cleaned',
            model: 'LGBModel',
            strategy: 'TopkDropoutStrategy',
            datasetRange: '2020-01-02 ~ 2026-07-01',
            conclusion: '清洗后数据 baseline 已跑通。',
            nextStep: '继续对比 canonical/cleaned。',
            metrics: {
              '1day.excess_return_with_cost.annualized_return': 0.198637,
              '1day.excess_return_with_cost.information_ratio': 1.705515,
              '1day.excess_return_with_cost.max_drawdown': -0.091497,
            },
          },
          {
            runId: 'abc123',
            experimentName: 'baseline_lgb_alpha158',
            status: 'completed',
            qlibProvider: '/Users/cq/WorkingProjects/quant_data/qlib/cn_data_baostock_jqdata',
            model: 'LGBModel',
            strategy: 'TopkDropoutStrategy',
            datasetRange: '2020-01-02 ~ 2026-07-01',
            conclusion: '长历史 baseline 已跑通，但收益质量偏弱。',
            nextStep: '补充风险控制和市场择时实验。',
            metrics: {
              '1day.excess_return_with_cost.annualized_return': 0.033970674,
              '1day.excess_return_with_cost.information_ratio': 0.261546,
            },
          },
        ],
      },
      externalTools: {
        tonghuashun: {
          name: '同花顺桌面端',
          status: 'available',
          installed: true,
          mode: 'read_only',
          path: '/Applications/同花顺.app',
          summary: '已检测到本机同花顺，可作为后续只读观察层。',
          capabilities: ['本机应用存在检测', '后续可接入只读窗口观察或截图/OCR'],
          restrictions: ['不读取账号密码', '不自动下单', '不连接券商交易接口'],
        },
        iphoneMirroring: {
          name: 'iPhone 镜像',
          status: 'deferred',
          installed: false,
          mode: 'deferred',
          path: null,
          summary: '手机连接暂时不做；未来也默认只做只读观察，不读取密码、不自动交易。',
          capabilities: [],
          restrictions: ['本阶段不接入', '不读取账号密码', '不自动下单'],
        },
      },
      capabilityMatrix: [
        {
          id: 'dsa_web_core',
          module: 'daily_stock_analysis WebUI',
          status: 'available',
          entry: '首页 / 研究 / 工作流 / 回测 / 问股 / 持仓 / 告警 / 选股',
          route: '/workflow',
          tests: '后端 pytest + 前端 vitest/lint/build',
          browser: 'local-workbench smoke 覆盖主页面',
          review: 'opencode复审通过',
          next: '继续把重任务体验做顺',
          restrictions: [],
        },
        {
          id: 'quant_workflow_pipeline',
          module: 'quant_workflow 流水线',
          status: 'available',
          entry: '工作流任务中心',
          route: '/workflow',
          tests: 'quant_workflow unittest + DSA quant API 测试',
          browser: '可从网页运行数据检查、质量报告、训练入口',
          review: 'opencode复审通过',
          next: '补更多产物预览和失败排查',
          restrictions: ['只允许白名单脚本'],
        },
        {
          id: 'qlib_integration',
          module: 'Qlib 机器学习量化',
          status: 'available',
          entry: '回测 / 工作流 / MLflow',
          route: '/backtest',
          tests: 'train_baseline_lgb + research_status API 测试',
          browser: '网页可查看 Qlib 训练摘要和回测入口',
          review: 'opencode复审通过',
          next: '继续补模型质量和清仓信号',
          restrictions: ['不直接改造 Qlib 官方源码为主 UI'],
        },
        {
          id: 'china_astock_quant_reuse',
          module: 'china-astock-quant',
          status: 'partial',
          entry: '研究页项目关系',
          route: '/research',
          tests: 'CodeGraph 已初始化，尚未直接包装脚本',
          browser: '暂作为策略参考展示',
          review: '待接入具体脚本后复审',
          next: '筛选可复用策略并白名单化',
          restrictions: ['不接入交易执行'],
        },
        {
          id: 'external_tools_tonghuashun',
          module: '同花顺桌面端',
          status: 'read_only',
          entry: '研究页 / 本机只读检测',
          route: '/research',
          tests: 'research_status API + ResearchPage 测试',
          browser: '已检测到本机同花顺；能力：本机应用存在检测',
          review: 'opencode复审通过',
          next: '只读观察，暂不接入交易',
          restrictions: ['不读取账号密码', '不自动下单', '不连接券商交易接口'],
        },
        {
          id: 'trading_execution',
          module: '交易执行',
          status: 'not_planned',
          entry: '无网页入口',
          tests: '不提供交易 API',
          browser: '页面只展示研究、回测和观察能力',
          review: '安全边界需持续复审',
          next: '保持人工交易，不做自动下单',
          restrictions: ['不自动下单', '不读取券商密码', '不连接券商交易接口'],
        },
      ],
      functionInventory: [
        {
          id: 'qlib_data_pipeline',
          area: 'Qlib 接入',
          name: '数据检查/转换/训练/回测',
          status: 'available',
          entry: '工作流 + 回测 + MLflow',
          route: '/workflow',
          owner: 'quant_workflow + Qlib',
          coverage: '网页可触发数据检查、Qlib 转换和训练。',
          gaps: '继续补模型质量解释。',
          next: '增加 Qlib 模型效果解释。',
          verification: 'train_baseline_lgb 最小测试。',
        },
        {
          id: 'china_astock_reference',
          area: '参考项目复用',
          name: 'china-astock-quant 可复用能力',
          status: 'partial',
          entry: '研究页清单 + 工作流对比报告',
          route: '/research',
          owner: 'china-astock-quant + quant_workflow',
          coverage: '已作为策略、指标、回测工程经验参考。',
          gaps: '尚未把策略直接白名单包装成可运行回测任务。',
          next: '优先接入指标库和策略对比。',
          verification: 'run_china_astock_comparison 测试。',
        },
        {
          id: 'quant_workflow_script_coverage',
          area: '量化流水线',
          name: 'quant_workflow 脚本覆盖盘点',
          status: 'partial',
          entry: '研究页功能入口盘点',
          route: '/research',
          owner: 'quant_workflow + daily_stock_analysis',
          coverage: '网页白名单覆盖核心任务；add_index_benchmark.py 与 generate_csi300.py 目前作为 provider 维护脚本保留在脚本层。',
          gaps: 'supervise_baostock.py 与网页任务中心的持久任务能力重叠，暂不单独暴露。',
          next: '把 provider 维护脚本改造成安全 dry-run 包装后，再考虑加入工作流任务中心。',
          verification: 'research_status API 测试 + ResearchPage 功能入口盘点展示。',
        },
      ],
      chinaAstockReuse: [
        {
          id: 'indicators',
          capability: '技术指标库',
          source: 'strategies/_indicators.py',
          status: 'available',
          priority: 'P1',
          decision: '优先复用',
          target: 'quant_workflow 指标/策略报告',
          next: '复用 sma/ema/rsi/atr/bollinger 做 DSA 技术分析和策略报告补强。',
          restrictions: [],
        },
        {
          id: 'paper_trading',
          capability: '纸面交易/券商适配',
          source: 'trading/*',
          status: 'not_planned',
          priority: '禁止项',
          decision: '本阶段不接入',
          target: '无网页入口',
          next: '保持为参考代码，不进入 DSA 主工作台。',
          restrictions: ['不自动下单', '不读取券商密码', '不连接券商交易接口'],
        },
      ],
      acceptanceChecklist: [
        {
          id: 'project_hygiene',
          title: '项目整洁',
          status: 'partial',
          route: '/research',
          evidence: 'DSA 无散落未跟踪源码；当前仍有已暂存正式改动。',
          next: '完成最终提交或明确保留暂存改动。',
        },
        {
          id: 'qlib_web_flow',
          title: 'Qlib 网页流程',
          status: 'available',
          route: '/workflow',
          evidence: '网页可触发数据检查、Qlib 转换、训练任务、日志、产物、回测摘要和 MLflow 链接。',
          next: '继续补模型效果解释。',
        },
        {
          id: 'opencode_review',
          title: 'opencode 只读复审',
          status: 'verified',
          route: '/research',
          evidence: '阶段改动已经过 opencode 子 agent 只读 review。',
          next: '每个阶段继续复审。',
        },
      ],
    });
    archiveExperimentRun.mockResolvedValue({
      removedRun: {
        runId: 'abc123',
        experimentName: 'baseline_lgb_alpha158',
        status: 'completed',
      },
      experimentNotebook: {
        status: 'available',
        totalRuns: 0,
        latestNote: '# 实验记录本',
        runs: [],
      },
    });
    permanentlyDeleteExperimentRun.mockResolvedValue({
      removedRun: {
        runId: 'abc123',
        experimentName: 'baseline_lgb_alpha158',
        status: 'completed',
      },
      experimentNotebook: {
        status: 'available',
        totalRuns: 0,
        latestNote: '# 实验记录本',
        runs: [],
      },
    });
  });

  it('renders the Chinese research workspace with six analysis categories and ETF boundaries', async () => {
    renderResearchPage('zh');

    expect(screen.getByRole('heading', { name: '研究工作台' })).toBeInTheDocument();
    expect(screen.getByText('基本面分析')).toBeInTheDocument();
    expect(screen.getByText('技术面/技术分析')).toBeInTheDocument();
    expect(screen.getByText('资金面分析')).toBeInTheDocument();
    expect(screen.getByText('情绪面/情绪分析')).toBeInTheDocument();
    expect(screen.getByText('宏观/市场复盘')).toBeInTheDocument();
    expect(screen.getByText('量化/模型实验')).toBeInTheDocument();

    expect(screen.getByText('场内基金/ETF分析')).toBeInTheDocument();
    expect(screen.getAllByText(/基本面部分降级/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/不按公司财报分析/).length).toBeGreaterThan(0);
    expect(screen.getByText('本机炒股软件读取')).toBeInTheDocument();
    expect(screen.getByText('iPhone 镜像 App 观察')).toBeInTheDocument();

    expect(screen.getByRole('link', { name: /打开 MLflow/ })).toHaveAttribute('href', 'http://127.0.0.1:6006/');
    expect(screen.getByRole('link', { name: /打开工作流/ })).toHaveAttribute('href', '/workflow');

    expect(await screen.findByText('数据状态')).toBeInTheDocument();
    expect(screen.getByText('可训练')).toBeInTheDocument();
    expect(screen.getByText('akshare')).toBeInTheDocument();
    expect(screen.getByText('jqdata')).toBeInTheDocument();
    expect(screen.getAllByText('部分可用').length).toBeGreaterThan(0);
    expect(screen.getAllByText('可用').length).toBeGreaterThan(0);
    expect(screen.queryByText('available')).not.toBeInTheDocument();
    expect(screen.getByText('最近训练结果')).toBeInTheDocument();
    expect(screen.getAllByText('baseline_lgb_alpha158_cleaned').length).toBeGreaterThan(0);
    expect(screen.getByText('cleaned123')).toBeInTheDocument();

    expect(screen.getByText('实验记录本')).toBeInTheDocument();
    expect(screen.getByText('最近 2 次实验')).toBeInTheDocument();
    expect(screen.getByText('清洗对比')).toBeInTheDocument();
    expect(screen.getByText('清洗后数据')).toBeInTheDocument();
    expect(screen.getByText('原始合并数据')).toBeInTheDocument();
    expect(screen.getByText('+0.1647')).toBeInTheDocument();
    expect(screen.getAllByText('LGBModel').length).toBeGreaterThan(0);
    expect(screen.getAllByText('TopkDropoutStrategy').length).toBeGreaterThan(0);
    expect(screen.getAllByText(/长历史 baseline 已跑通/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/补充风险控制和市场择时实验/).length).toBeGreaterThan(0);

    expect(screen.getByText('能力验收矩阵')).toBeInTheDocument();
    expect(screen.getAllByText('daily_stock_analysis WebUI').length).toBeGreaterThan(0);
    expect(screen.getAllByText('quant_workflow 流水线').length).toBeGreaterThan(0);
    expect(screen.getAllByText('Qlib 机器学习量化').length).toBeGreaterThan(0);
    expect(screen.getAllByText('china-astock-quant').length).toBeGreaterThan(0);
    expect(screen.getAllByText('同花顺桌面端').length).toBeGreaterThan(0);
    expect(screen.getAllByText('交易执行').length).toBeGreaterThan(0);
    expect(screen.getAllByText(/已检测到本机同花顺/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/本机应用存在检测/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/不自动下单/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/不读取券商密码/).length).toBeGreaterThan(0);
    expect(screen.getAllByText('测试脚本').length).toBeGreaterThan(0);
    expect(screen.getAllByText('浏览器执行').length).toBeGreaterThan(0);
    expect(screen.getAllByText('opencode复审').length).toBeGreaterThan(0);
    expect(screen.getByText('功能入口盘点')).toBeInTheDocument();
    expect(screen.getAllByText('数据检查/转换/训练/回测').length).toBeGreaterThan(0);
    expect(screen.getAllByText('train_baseline_lgb 最小测试。').length).toBeGreaterThan(0);
    expect(screen.getAllByText('quant_workflow 脚本覆盖盘点').length).toBeGreaterThan(0);
    expect(screen.getAllByText(/add_index_benchmark\.py/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/generate_csi300\.py/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/supervise_baostock\.py/).length).toBeGreaterThan(0);
    expect(screen.getByText('china-astock-quant 复用清单')).toBeInTheDocument();
    expect(screen.getByText('技术指标库')).toBeInTheDocument();
    expect(screen.getByText('strategies/_indicators.py')).toBeInTheDocument();
    expect(screen.getByText('纸面交易/券商适配')).toBeInTheDocument();
    expect(screen.getByText('阶段验收清单')).toBeInTheDocument();
    expect(screen.getByText('项目整洁')).toBeInTheDocument();
    expect(screen.getAllByText(/无散落未跟踪源码/).length).toBeGreaterThan(0);
    expect(screen.getAllByText('Qlib 网页流程').length).toBeGreaterThan(0);
    expect(screen.getAllByText(/MLflow 链接/).length).toBeGreaterThan(0);
    expect(screen.getByText('opencode 只读复审')).toBeInTheDocument();
    expect(screen.getByText('后续开发队列')).toBeInTheDocument();
    expect(screen.getByText('每日研究向导')).toBeInTheDocument();
    expect(screen.getByText('模型质量和清仓信号')).toBeInTheDocument();
    expect(screen.getByText('场内基金/ETF深度版')).toBeInTheDocument();
    expect(screen.getByText('同花顺只读观察层')).toBeInTheDocument();
    expect(screen.getAllByText('P1').length).toBeGreaterThan(0);
    expect(screen.getAllByRole('link', { name: '工作流 / 今日流程' }).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/不读取账号密码/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/不连接券商交易接口/).length).toBeGreaterThan(0);
  });

  it('renders English copy without missing i18n keys', async () => {
    renderResearchPage('en');

    expect(screen.getByRole('heading', { name: 'Research Workspace' })).toBeInTheDocument();
    expect(screen.getByText('Experiment Notebook')).toBeInTheDocument();
    expect(await screen.findByText('Latest 2 experiments')).toBeInTheDocument();
    expect(screen.getByText('Cleaned vs Canonical')).toBeInTheDocument();
    expect(screen.getByText('Exchange-traded fund / ETF analysis')).toBeInTheDocument();
    expect(screen.getAllByText(/company financial statements are downgraded/).length).toBeGreaterThan(0);
    expect(await screen.findByText('Data Status')).toBeInTheDocument();
    expect(screen.getByText('Trainable')).toBeInTheDocument();
    expect(screen.getAllByText('Partial').length).toBeGreaterThan(0);
    expect(screen.getAllByText('Available').length).toBeGreaterThan(0);
    expect(screen.getByText('Capability Acceptance Matrix')).toBeInTheDocument();
    expect(screen.getByText('Function Entry Inventory')).toBeInTheDocument();
    expect(screen.getByText('china-astock-quant Reuse List')).toBeInTheDocument();
    expect(screen.getByText('Acceptance Checklist')).toBeInTheDocument();
    expect(screen.getByText('Development Queue')).toBeInTheDocument();
    expect(screen.getByText('Daily research guide')).toBeInTheDocument();
    expect(screen.getByText('Model quality and cash signal')).toBeInTheDocument();
    expect(screen.getAllByText('Automated tests').length).toBeGreaterThan(0);
    expect(screen.getAllByText('Browser execution').length).toBeGreaterThan(0);
    expect(screen.getAllByText('opencode review').length).toBeGreaterThan(0);
    expect(screen.queryByText(/research\./)).not.toBeInTheDocument();
  });

  it('falls back to static coverage items when the capability matrix is not returned', async () => {
    getResearchStatus.mockResolvedValueOnce({
      generatedAt: '2026-07-02T10:00:00+08:00',
      trainable: false,
      trainingReadiness: {
        status: 'missing',
        summary: 'Research status artifact is not available yet.',
      },
      dataSources: [],
      qlibProvider: null,
      latestTraining: null,
      experimentNotebook: {
        status: 'missing',
        totalRuns: 0,
        latestNote: '',
        runs: [],
      },
      externalTools: {
        tonghuashun: {
          name: '同花顺桌面端',
          status: 'missing',
          installed: false,
          mode: 'read_only',
          path: null,
          summary: '未在默认路径检测到同花顺。',
          capabilities: [],
          restrictions: ['不读取账号密码', '不自动下单'],
        },
        iphoneMirroring: {
          name: 'iPhone 镜像',
          status: 'deferred',
          installed: false,
          mode: 'deferred',
          path: null,
          summary: '手机连接暂时不做。',
          capabilities: [],
          restrictions: ['本阶段不接入'],
        },
      },
      capabilityMatrix: null,
    });

    renderResearchPage('zh');

    expect(await screen.findByText('能力验收矩阵')).toBeInTheDocument();
    expect(screen.getAllByText('quant_workflow + Qlib').length).toBeGreaterThan(0);
    expect(screen.getAllByText('iPhone 镜像').length).toBeGreaterThan(0);
    expect(screen.getAllByText(/手机连接暂时不做/).length).toBeGreaterThan(0);
  });

  it('archives a selected experiment run from the notebook list', async () => {
    renderResearchPage('zh');

    fireEvent.click(await screen.findByRole('button', { name: '归档实验 abc123' }));

    await waitFor(() => {
      expect(archiveExperimentRun).toHaveBeenCalledWith('abc123');
    });
    expect(screen.queryByText('TopkDropoutStrategy')).not.toBeInTheDocument();
    expect(screen.getByText('还没有结构化实验条目。')).toBeInTheDocument();
  });

  it('permanently deletes a selected experiment run after confirmation', async () => {
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true);
    renderResearchPage('zh');

    fireEvent.click(await screen.findByRole('button', { name: '永久删除实验 abc123' }));

    await waitFor(() => {
      expect(permanentlyDeleteExperimentRun).toHaveBeenCalledWith('abc123');
    });
    expect(confirmSpy).toHaveBeenCalled();
    expect(screen.queryByText('TopkDropoutStrategy')).not.toBeInTheDocument();
    confirmSpy.mockRestore();
  });
});
