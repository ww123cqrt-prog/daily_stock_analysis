import { test, expect } from '@playwright/test';
import type { ConsoleMessage, Page } from '@playwright/test';

const localBaseUrl = process.env.DSA_WEB_E2E_BASE_URL || 'http://127.0.0.1:8010';

test.use({
  baseURL: localBaseUrl,
  locale: 'zh-CN',
});

function watchConsole(page: Page) {
  const errors: string[] = [];
  const onConsole = (message: ConsoleMessage) => {
    if (message.type() === 'error') {
      errors.push(message.text());
    }
  };
  const onPageError = (error: Error) => {
    errors.push(error.message);
  };
  page.on('console', onConsole);
  page.on('pageerror', onPageError);
  return {
    errors,
    stop: () => {
      page.off('console', onConsole);
      page.off('pageerror', onPageError);
    },
  };
}

test.describe('Research Page - Data Hygiene', () => {
  test('should display data hygiene status cards', async ({ page }) => {
    const consoleWatch = watchConsole(page);
    try {
      await page.goto('/research');
      await page.waitForLoadState('domcontentloaded');

      // Check page title
      await expect(page).toHaveTitle(/研究工作台/);

      // Check if data status section exists
      const dataStatusSection = page.getByRole('region', { name: /数据状态/i });
      await expect(dataStatusSection).toBeVisible();

      // Check if data hygiene section exists and verify its content.
      const dataHygieneSection = page.getByRole('region', { name: /数据治理/i });
      await expect(dataHygieneSection).toBeVisible({ timeout: 15_000 });

      console.log('✓ Data hygiene section is visible');

      // Check maintenance lock badge
      const maintenanceLockBadge = page.getByText(/维护锁: (未锁定|已锁定)/);
      await expect(maintenanceLockBadge).toBeVisible();
      console.log('✓ Maintenance lock status displayed');

      const manifestStatus = dataHygieneSection.getByText(/数据 Manifest: (可用|已过期|缺失)/);
      await expect(manifestStatus).toBeVisible();
      if ((await manifestStatus.textContent())?.includes('已过期')) {
        await expect(
          dataHygieneSection.getByText('盘点与当前数据清单不一致，请重新扫描。'),
        ).toBeVisible();
      }
      await expect(page.getByText(/治理说明: 可用/)).toBeVisible();
      console.log('✓ Data governance manifest and README status displayed');

      // Check for inventory card
      const inventoryCard = page.getByText(/数据盘点/).first();
      await expect(inventoryCard).toBeVisible();
      console.log('✓ Inventory card displayed');

      // Check for cleanup plan card
      const cleanupPlanCard = page.getByText(/清理计划/).first();
      await expect(cleanupPlanCard).toBeVisible();
      console.log('✓ Cleanup plan card displayed');

      // Check for quality report card
      const qualityReportCard = page.getByText(/质量报告/).first();
      await expect(qualityReportCard).toBeVisible();
      console.log('✓ Quality report card displayed');

      // Check for cleaning strategy card
      const cleaningStrategyCard = page.getByText(/清洗策略/).first();
      await expect(cleaningStrategyCard).toBeVisible();
      console.log('✓ Cleaning strategy card displayed');

      // Take screenshot of data hygiene section
      await page.screenshot({
        path: 'test-results/data-hygiene-section.png',
        fullPage: false,
      });
      console.log('✓ Screenshot saved to test-results/data-hygiene-section.png');
      expect(consoleWatch.errors).toEqual([]);
    } finally {
      consoleWatch.stop();
    }
  });

  test('should display correct status badges', async ({ page }) => {
    const consoleWatch = watchConsole(page);
    try {
      await page.goto('/research');
      await page.waitForLoadState('domcontentloaded');

      const dataHygieneSection = page.getByRole('region', { name: /数据治理/i });
      await expect(dataHygieneSection).toBeVisible({ timeout: 15_000 });

      // Check for visible status labels without depending on a specific CSS class.
      const statusLabels = dataHygieneSection.getByText(/可用|已过期|缺失|未锁定|已锁定/);
      await expect(statusLabels.first()).toBeVisible();
      const statusCount = await statusLabels.count();
      expect(statusCount).toBeGreaterThanOrEqual(3);
      console.log(`✓ Found ${statusCount} status labels`);

      for (const label of ['数据盘点', '清理计划', '质量报告', '清洗策略']) {
        await expect(dataHygieneSection.getByText(label).first()).toBeVisible();
      }
      expect(consoleWatch.errors).toEqual([]);
    } finally {
      consoleWatch.stop();
    }
  });
});
