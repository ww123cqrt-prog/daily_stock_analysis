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
  const toolbarBox = await page.locator('#dsa-mlflow-language-toolbar').boundingBox();
  const mlflowRootBox = await page.locator('#root.mlflow-ui-container').boundingBox();
  expect(toolbarBox).not.toBeNull();
  expect(mlflowRootBox).not.toBeNull();
  expect(mlflowRootBox!.y).toBeGreaterThanOrEqual(toolbarBox!.y + toolbarBox!.height - 1);
  await expect(page.getByRole('button', { name: '中文' })).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByRole('link', { name: '实验', exact: true })).toBeVisible();
  await expect(page.getByText('GenAI', { exact: true }).first()).toBeVisible();

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

test('keeps MLflow business data verbatim and preserves the complete upstream navigation', async ({ page }) => {
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
    <div data-natural-counter><span>4</span><span data-natural-slash title="/">/</span><span>17</span></div>
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
  await expect(page.locator('a[href="#/"]')).toBeVisible();
  await expect(page.locator('a[href="#/prompts"]')).toBeVisible();
  await expect(page.locator('a[href="#/gateway"]')).toBeVisible();
  await expect(page.locator('[data-nonquant="genai"]')).toBeVisible();
  await expect(page.locator('[data-nonquant="assistant"]')).toBeVisible();
  await expect(page.locator('[data-nonquant="docs"]')).toBeVisible();

  await page.evaluate(() => window.scrollTo(0, 700));
  const scrollBefore = await page.evaluate(() => window.scrollY);
  await page.getByRole('button', { name: 'English' }).click();
  expect(await page.evaluate(() => window.scrollY)).toBe(scrollBefore);
  await expect(page.locator('[data-natural-counter]')).toHaveText('4/17');
  await expect(page.locator('[data-natural-slash]')).toHaveAttribute('title', '/');

  await expect(page.getByRole('link', { name: 'DSA', exact: true })).toHaveAttribute('href', '/research');

  await page.evaluate(() => {
    window.location.hash = '#/gateway';
    window.dispatchEvent(new HashChangeEvent('hashchange'));
  });
  await expect.poll(() => page.evaluate(() => window.location.hash)).toBe('#/gateway');
});

test('preserves run filters, parameter names, and artifacts on the local MLflow data', async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('dsa.uiLanguage', 'en'));
  await page.goto('/experiments/');
  await page.getByRole('button', { name: 'Model training', exact: true }).click();
  await expect(page.getByRole('link', { name: 'Experiments', exact: true })).toBeVisible();

  const experimentLinks = page.locator('[data-testid="experiment-list-item-link"]');
  await expect(experimentLinks.first()).toBeVisible();
  const experimentCandidates = await experimentLinks.evaluateAll((links) => (
    links.map((link) => ({
      href: link.getAttribute('href') || '',
      name: (link.textContent || '').trim(),
    })).filter((candidate) => candidate.href && candidate.name)
  ));
  expect(experimentCandidates.length).toBeGreaterThan(0);

  let experimentName = '';
  for (const candidate of experimentCandidates) {
    await page.goto(new URL(candidate.href, page.url()).toString());
    const hasRun = await page.locator('a[href*="/runs/"]').first().waitFor({
      state: 'visible',
      timeout: 5_000,
    }).then(() => true).catch(() => false);
    if (hasRun) {
      experimentName = candidate.name;
      break;
    }
  }
  expect(experimentName, 'Expected at least one local MLflow experiment containing a run.').not.toBe('');
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
