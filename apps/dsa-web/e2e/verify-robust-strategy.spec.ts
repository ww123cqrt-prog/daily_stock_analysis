import { expect, test, type Page } from '@playwright/test';

const researchStatus = {
  generated_at: '2026-07-16T18:00:00+08:00',
  trainable: true,
  training_readiness: { status: 'trainable', summary: 'Robust provider is ready.' },
  data_sources: [],
  qlib_provider: {
    path: 'qlib/robust_provider',
    calendar_days: 3000,
    calendar_range: '2014-01-02 ~ 2026-07-16',
    instruments: 800,
  },
  robust_strategy: {
    status: 'shadow_observation',
    lifecycle_status: 'shadow_observation',
    run_id: 'confirmation-2026',
    updated_at: '2026-07-16T17:55:00+08:00',
    stages: [
      { name: 'build_pit_csi800', status: 'completed' },
      { name: 'run_robust_development', status: 'completed' },
      { name: 'run_robust_confirmation', status: 'completed' },
    ],
    preflight: { status: 'completed' },
    development: {
      status: 'research_candidate_ready',
      selected_models: ['linear_ridge', 'lightgbm'],
      selected_strategy: 'full_risk_control',
      model_stability: [
        {
          model_id: 'linear_ridge',
          eligible: true,
          stability: { rank_correlation_median: 0.76, topk_overlap_mean: 0.65 },
        },
        {
          model_id: 'lightgbm',
          eligible: true,
          stability: { rank_correlation_median: 0.74, topk_overlap_mean: 0.63 },
        },
      ],
      strategy_matrix: [
        {
          strategy_id: 'full_risk_control',
          acceptance: { passed: true },
          baseline: { metrics: { annual_excess_return: 0.08 } },
          stress: { metrics: { annual_excess_return: 0.02 } },
          delayed: { metrics: { annual_excess_return: 0.01 } },
          quarter_metrics: [{ quarter: '2025Q4', annual_excess_return: 0.05, rank_ic: 0.03 }],
          regime_metrics: { risk_on: { annual_excess_return: 0.09, days: 120 } },
        },
      ],
    },
    confirmation: {
      status: 'shadow_observation',
      latest_market_day: '2026-07-15',
      metrics: { annual_excess_return: 0.03, information_ratio: 0.4 },
      verdict: { passed: true },
    },
    shadow: {
      status: 'shadow_observation',
      metrics: { trading_days: 3, net_excess_return: 0.01, task_success_ratio: 0.67 },
      verdict: { passed: false, remaining_days: 57 },
    },
    shadow_daily: {
      status: 'development_failed',
      reason_code: 'inference_failed',
      as_of: '2026-07-16',
      inference_success: false,
      appended_days: 1,
    },
    mlflow_experiments: ['robust_csi800_development', 'robust_csi800_confirmation'],
  },
  mlflow: { url: '/experiments/', status: 'available' },
};

const quantStatus = {
  available_job_types: {
    run_robust_strategy: {
      label: '运行鲁棒策略开发流程',
      allowed_params: ['as_of_date', 'run_id', 'resume', 'skip_membership_build'],
    },
    run_robust_confirmation: {
      label: '运行冻结策略 2026 确认期',
      allowed_params: ['run_id', 'resume'],
    },
    run_robust_shadow_day: {
      label: '运行鲁棒策略当日纸面跟踪',
      allowed_params: ['as_of_date'],
    },
  },
  active_count: 0,
  latest_jobs: {},
  recent_jobs: [],
  mlflow: { url: '/experiments/', status: 'available' },
  external_services: {},
};

async function installRobustRoutes(page: Page) {
  await page.addInitScript(() => window.localStorage.setItem('dsa_ui_language', 'zh'));
  await page.route('**/api/v1/research/status', async (route) => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(researchStatus) });
  });
  await page.route('**/api/v1/quant/status', async (route) => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(quantStatus) });
  });
}

test('keeps lifecycle and latest shadow task separate across workflow and backtest', async ({ page }) => {
  await installRobustRoutes(page);
  await page.goto('/workflow');

  await expect(page.getByRole('heading', { name: '鲁棒策略生命周期' })).toBeVisible();
  await expect(page.getByText('60日纸面跟踪中')).toBeVisible();
  await expect(page.getByTestId('robust-shadow-daily')).toContainText('最近纸面任务');
  await expect(page.getByTestId('robust-shadow-daily')).toContainText('2026-07-16');
  await expect(page.getByTestId('robust-shadow-daily')).toContainText('inference_failed');

  await page.getByRole('link', { name: '查看鲁棒性矩阵' }).click();
  await expect(page).toHaveURL(/\/backtest$/);
  await page.getByRole('button', { name: 'Qlib模型/策略回测' }).click();

  await expect(page.getByRole('heading', { name: '鲁棒策略矩阵' })).toBeVisible();
  await expect(page.getByText('最近纸面任务失败')).toBeVisible();
  await expect(page.getByText(/2026-07-16.*inference_failed/)).toBeVisible();
  await expect(page.getByText('纸面跟踪 3 / 60 日')).toBeVisible();
});

test('robust workflow remains readable without page-level overflow on mobile', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await installRobustRoutes(page);
  await page.goto('/workflow');

  await expect(page.getByRole('heading', { name: '鲁棒策略生命周期' })).toBeVisible();
  await expect(page.getByTestId('robust-shadow-daily')).toContainText('inference_failed');
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow).toBeLessThanOrEqual(1);
});
