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
    expect(screen.getByText('技术分析')).toBeInTheDocument();
    expect(screen.getByText('资金面分析')).toBeInTheDocument();
    expect(screen.getByText('情绪分析')).toBeInTheDocument();
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

    expect(screen.getByText('自助覆盖矩阵')).toBeInTheDocument();
    expect(screen.getAllByText('daily_stock_analysis WebUI').length).toBeGreaterThan(0);
    expect(screen.getAllByText('quant_workflow + Qlib').length).toBeGreaterThan(0);
    expect(screen.getAllByText('china-astock-quant').length).toBeGreaterThan(0);
    expect(screen.getAllByText('同花顺桌面端').length).toBeGreaterThan(0);
    expect(screen.getAllByText(/已检测到本机同花顺/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/本机应用存在检测/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/不自动下单/).length).toBeGreaterThan(0);
    expect(screen.getAllByText('测试脚本').length).toBeGreaterThan(0);
    expect(screen.getAllByText('浏览器执行').length).toBeGreaterThan(0);
    expect(screen.getAllByText('opencode复审').length).toBeGreaterThan(0);
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
    expect(screen.getByText('Self-service Coverage Matrix')).toBeInTheDocument();
    expect(screen.getByText('Development Queue')).toBeInTheDocument();
    expect(screen.getByText('Daily research guide')).toBeInTheDocument();
    expect(screen.getByText('Model quality and cash signal')).toBeInTheDocument();
    expect(screen.getAllByText('Automated tests').length).toBeGreaterThan(0);
    expect(screen.getAllByText('Browser execution').length).toBeGreaterThan(0);
    expect(screen.getAllByText('opencode review').length).toBeGreaterThan(0);
    expect(screen.queryByText(/research\./)).not.toBeInTheDocument();
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
