import { test, expect } from '@playwright/test';

const localBaseUrl = process.env.DSA_WEB_E2E_BASE_URL || 'http://127.0.0.1:18010';

test.use({
  baseURL: localBaseUrl,
  locale: 'zh-CN',
});

test('shows the five-model candidate pipeline controls', async ({ page }) => {
  const consoleErrors: string[] = [];
  page.on('console', (message) => {
    if (message.type() === 'error') consoleErrors.push(message.text());
  });
  page.on('pageerror', (error) => consoleErrors.push(error.message));

  await page.goto('/workflow');
  await expect(page.getByText('运行五模型完整量化流程', { exact: true })).toBeVisible();
  await expect(page.getByLabel('运行五模型完整量化流程 数据版本')).toBeVisible();
  await expect(page.getByLabel('运行五模型完整量化流程 只做小样本预演')).toBeVisible();
  await expect(page.getByRole('button', { name: '运行 运行五模型完整量化流程' })).toBeVisible();
  expect(consoleErrors).toEqual([]);
});
