import { expect, test } from '@playwright/test';

const localBaseUrl = process.env.DSA_WEB_E2E_BASE_URL || 'http://127.0.0.1:8010';

test.use({
  baseURL: localBaseUrl,
  locale: 'zh-CN',
});

test('serves the official MLflow UI under DSA and preserves language across reloads', async ({ page }) => {
  await page.addInitScript(() => {
    if (localStorage.getItem('dsa.uiLanguage') === null) {
      localStorage.setItem('dsa.uiLanguage', 'zh');
      localStorage.removeItem('mlflow-ui-language');
    }
  });

  await page.goto('/experiments/');
  await expect(page).toHaveTitle(/MLflow/);
  await expect(page.getByRole('button', { name: '中文' })).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByRole('link', { name: '实验', exact: true })).toBeVisible();
  await expect(page.getByText('GenAI', { exact: true })).toBeHidden();

  const originalUrl = page.url();
  await page.getByRole('button', { name: 'English' }).click();
  await expect(page.getByRole('link', { name: 'Experiments', exact: true })).toBeVisible();
  expect(page.url()).toBe(originalUrl);

  await page.reload();
  await expect(page.getByRole('button', { name: 'English' })).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByRole('link', { name: 'Experiments', exact: true })).toBeVisible();

  await page.getByRole('button', { name: '中文' }).click();
  await expect(page.getByRole('link', { name: '实验', exact: true })).toBeVisible();
});

test('keeps MLflow business data verbatim and limits the page to experiment workflows', async ({ page }) => {
  await page.goto('/experiments/#/experiments');
  const runtimeResponse = await page.request.get('/experiments/__dsa_i18n__/runtime.js');
  expect(runtimeResponse.ok()).toBe(true);
  const runtime = await runtimeResponse.text();

  await page.setContent(`
    <nav>
      <a href="#/">Home</a>
      <a href="#/experiments">Experiments</a>
      <a href="#/prompts">Prompts</a>
      <a href="#/gateway">AI Gateway</a>
      <div role="button" data-nonquant="genai">GenAI</div>
      <div role="button" data-nonquant="assistant">Assistant <span>Beta</span></div>
      <a href="https://www.mlflow.org/docs/latest/genai/" data-nonquant="docs">Docs</a>
    </nav>
    <div role="columnheader" data-ui-label>Model</div>
    <button type="button" data-ui-action>Status</button>
    <div role="cell" data-business="experiment">Home</div>
    <div role="gridcell" data-business="run">Model</div>
    <div role="cell" data-business="parameter">Status</div>
    <div role="cell" data-business="metric">Home</div>
    <div data-testid="artifact-tree-node" artifact-name="Model" data-business="artifact">Model</div>
    <div data-testid="key-value-Home">
      <span data-business="key-value-key">Home</span>
      <span data-business="key-value-value">Model</span>
    </div>
    <div style="height: 2000px"></div>
  `);
  await page.addScriptTag({ content: runtime });

  await expect(page.locator('[data-ui-label]')).toHaveText('模型');
  await expect(page.locator('[data-ui-action]')).toHaveText('状态');
  await expect(page.locator('[data-business="experiment"]')).toHaveText('Home');
  await expect(page.locator('[data-business="run"]')).toHaveText('Model');
  await expect(page.locator('[data-business="parameter"]')).toHaveText('Status');
  await expect(page.locator('[data-business="metric"]')).toHaveText('Home');
  await expect(page.locator('[data-business="artifact"]')).toHaveText('Model');
  await expect(page.locator('[data-business="key-value-key"]')).toHaveText('Home');
  await expect(page.locator('[data-business="key-value-value"]')).toHaveText('Model');
  await expect(page.locator('a[href="#/prompts"]')).toBeHidden();
  await expect(page.locator('a[href="#/gateway"]')).toBeHidden();
  await expect(page.locator('[data-nonquant="genai"]')).toBeHidden();
  await expect(page.locator('[data-nonquant="assistant"]')).toBeHidden();
  await expect(page.locator('[data-nonquant="docs"]')).toBeHidden();

  await page.evaluate(() => window.scrollTo(0, 700));
  const scrollBefore = await page.evaluate(() => window.scrollY);
  await page.getByRole('button', { name: 'English' }).click();
  expect(await page.evaluate(() => window.scrollY)).toBe(scrollBefore);

  await expect(page.getByRole('link', { name: 'Research' })).toHaveAttribute('href', '/research');
  await expect(page.getByRole('link', { name: 'Workflow' })).toHaveAttribute('href', '/workflow');
  await expect(page.getByRole('link', { name: 'Backtest' })).toHaveAttribute('href', '/backtest');

  await page.evaluate(() => {
    window.location.hash = '#/gateway';
    window.dispatchEvent(new HashChangeEvent('hashchange'));
  });
  await expect.poll(() => page.evaluate(() => window.location.hash)).toBe('#/experiments');
});

test('preserves run filters, parameter names, and artifacts on the local MLflow data', async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('dsa.uiLanguage', 'en'));
  await page.goto('/experiments/');

  const firstExperiment = page.locator('[data-testid="experiment-list-item-link"]').first();
  await expect(firstExperiment).toBeVisible();
  await expect(firstExperiment).toHaveAttribute('href', /#\/experiments\/[^/]+\/runs$/);
  const experimentName = (await firstExperiment.innerText()).trim();
  await firstExperiment.click();
  await expect(page).toHaveURL(/#\/experiments\/[^/]+\/runs/);

  const search = page.locator('[data-testid="search-box"]');
  const filter = 'attributes.run_name = "__dsa_no_matching_run__"';
  await search.fill(filter);
  await search.press('Enter');
  await page.getByRole('button', { name: '中文' }).click();
  await expect(search).toHaveValue(filter);
  await search.fill('');
  await search.press('Enter');

  const firstRun = page.locator('a[href*="/runs/"]').first();
  await expect(firstRun).toBeVisible();
  const runName = (await firstRun.innerText()).trim();
  await firstRun.click();

  const parameterKey = page.getByText('model.class', { exact: true });
  await expect(parameterKey).toBeVisible();
  await expect(page.getByText(experimentName, { exact: true }).first()).toBeVisible();
  await expect(page.getByText(runName, { exact: true }).first()).toBeVisible();
  await page.getByRole('button', { name: 'English' }).click();
  await expect(parameterKey).toHaveText('model.class');
  await expect(page.getByText(experimentName, { exact: true }).first()).toBeVisible();
  await expect(page.getByText(runName, { exact: true }).first()).toBeVisible();

  await page.getByText('Artifacts', { exact: true }).click();
  const firstArtifact = page.locator('[data-testid="artifact-tree-node"]').first();
  await expect(firstArtifact).toBeVisible();
  const artifactName = (await firstArtifact.getAttribute('artifact-name')) || '';
  expect(artifactName).not.toBe('');
  await page.getByRole('button', { name: '中文' }).click();
  await expect(firstArtifact).toHaveAttribute('artifact-name', artifactName);
  await expect(firstArtifact).toContainText(artifactName);
});

test('keeps the experiment list readable on a mobile viewport', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.addInitScript(() => localStorage.setItem('dsa.uiLanguage', 'zh'));
  await page.goto('/experiments/');

  await expect(page.getByRole('heading', { name: '实验', exact: true })).toBeVisible();
  await expect(page.locator('aside').first()).toBeHidden();

  const table = page.getByRole('table').first();
  await expect(table).toBeVisible();
  const tableBox = await table.boundingBox();
  expect(tableBox).not.toBeNull();
  expect(tableBox!.width).toBeGreaterThanOrEqual(340);

  const toolbar = page.locator('#dsa-mlflow-language-toolbar');
  await expect(toolbar).toBeVisible();
  const toolbarBox = await toolbar.boundingBox();
  expect(toolbarBox).not.toBeNull();
  expect(toolbarBox!.y).toBeGreaterThan(700);
  expect(toolbarBox!.y + toolbarBox!.height).toBeLessThanOrEqual(840);
});
