import { expect, test, type Page } from '@playwright/test';

const baseURL = process.env.DSA_WEB_E2E_BASE_URL || 'http://127.0.0.1:8010';

test.use({ baseURL, locale: 'zh-CN' });

async function openChinesePage(page: Page, path: string) {
  await page.addInitScript(() => {
    window.localStorage.setItem('dsa.uiLanguage', 'zh');
  });
  await page.goto(path, { waitUntil: 'domcontentloaded' });
}

async function expectNoHorizontalOverflow(page: Page) {
  await expect.poll(async () => page.evaluate(() => (
    document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1
  ))).toBe(true);
}

test('separates data, training and other tasks and injects a training result into backtesting', async ({ page }, testInfo) => {
  await openChinesePage(page, '/workflow');
  await expect(page.getByRole('heading', { name: '选股模型训练' })).toBeVisible();
  await expect(page.locator('[data-job-type="train_baseline_lgb"]')).toBeVisible();
  await expect(page.locator('[data-job-type="run_candidate_pipeline"]')).toBeVisible();
  await expect(page.locator('[data-job-type="run_daily_update"]')).toHaveCount(0);
  await expect(page.locator('[data-job-type="report_qlib_predictions"]')).toHaveCount(0);
  await expect(page.getByText('Latest Training', { exact: true })).toHaveCount(0);
  await expect(page.getByText('最近训练', { exact: true })).toHaveCount(0);
  await expect(page.getByRole('link', { name: '实验记录' })).toHaveCount(0);
  await expect(page.getByRole('heading', { name: '训练选股模型' })).toBeVisible();
  await expect(page.getByRole('heading', { name: '已完成训练与策略结果' })).toBeVisible();

  const latestBacktest = page.getByRole('link', { name: '回测 baseline_lgb_alpha158 Qlib 默认组合' });
  await expect(latestBacktest).toBeVisible();
  await expectNoHorizontalOverflow(page);
  await page.screenshot({ path: testInfo.outputPath('workflow-desktop.png'), fullPage: true });
  await latestBacktest.click();
  await expect(page).toHaveURL(/\/backtest\?tab=trained&runId=.*&modelId=latest&strategyId=qlib_default/);
  await expect(page.getByRole('tab', { name: '训练结果策略回测' })).toHaveAttribute('aria-selected', 'true');
  await expect(page.getByLabel('训练模型')).toHaveValue('latest');
  await expect(page.getByLabel('交易策略')).toHaveValue('qlib_default');

  await openChinesePage(page, '/data');
  await expect(page.getByRole('heading', { name: '数据更新与检查' })).toBeVisible();
  await expect(page.locator('[data-job-type="run_daily_update"]')).toBeVisible();
  await expect(page.locator('[data-job-type="train_baseline_lgb"]')).toHaveCount(0);

  await openChinesePage(page, '/other');
  await expect(page.getByRole('heading', { name: '其他量化任务' })).toBeVisible();
  await expect(page.locator('[data-job-type="report_qlib_predictions"]')).toBeVisible();
  await expect(page.locator('[data-job-type="run_daily_update"]')).toHaveCount(0);
  await page.screenshot({ path: testInfo.outputPath('other-desktop.png'), fullPage: false });
});

test('keeps workflow and trained-result backtesting usable at mobile width', async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await openChinesePage(page, '/workflow');
  await expect(page.getByRole('heading', { name: '选股模型训练' })).toBeVisible();
  await expectNoHorizontalOverflow(page);

  const href = await page.getByRole('link', { name: '回测 baseline_lgb_alpha158 Qlib 默认组合' }).getAttribute('href');
  expect(href).toBeTruthy();
  await openChinesePage(page, href!);
  await expect(page.getByRole('tab', { name: '训练结果策略回测' })).toHaveAttribute('aria-selected', 'true');
  await expect(page.getByRole('heading', { name: '已训练模型与策略' })).toBeVisible();
  await expect(page.getByLabel('训练模型')).toHaveValue('latest');
  await expect(page.getByLabel('交易策略')).toHaveValue('qlib_default');
  await expect(page.getByText('正在读取训练结果')).toHaveCount(0);
  await expectNoHorizontalOverflow(page);
  await page.screenshot({ path: testInfo.outputPath('trained-backtest-mobile.png'), fullPage: false });
});
