import { defineConfig, devices } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const currentDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(currentDir, '../..');
const shouldManageWebServers = process.env.DSA_WEB_SMOKE_MANAGED === '1' || !!process.env.DSA_WEB_SMOKE_PASSWORD;
const backendPort = process.env.DSA_WEB_SMOKE_BACKEND_PORT || '8000';
const managedSmokeRoot = path.join(currentDir, 'test-results', 'managed-smoke');
const managedReportsRoot = process.env.DSA_WEB_SMOKE_REPORTS_ROOT
  || path.join(managedSmokeRoot, 'quant_workflow', 'reports');
const realReportsRoot = path.resolve(repoRoot, 'quant_workflow', 'reports');
const managedPythonFixtureRoot = path.join(currentDir, 'e2e', 'fixtures', 'python');

function copyIfExists(source: string, destination: string) {
  if (!fs.existsSync(source)) return;
  fs.mkdirSync(path.dirname(destination), { recursive: true });
  fs.copyFileSync(source, destination);
}

function seedManagedSmokeReports() {
  if (!shouldManageWebServers) return;

  fs.rmSync(managedSmokeRoot, { recursive: true, force: true });
  fs.mkdirSync(managedReportsRoot, { recursive: true });

  for (const fileName of [
    'research_status_latest.json',
    'latest_training_result.json',
    'train_baseline_lgb_status_latest.json',
    'experiment_runs.jsonl',
    'experiment_runs_deleted.jsonl',
    'experiment_log.md',
    'qlib_predictions_latest.json',
    'qlib_predictions_latest.md',
    'cash_control_backtest_latest.json',
    'cash_control_backtest_latest.md',
    'market_regime_latest.json',
    'market_regime_latest.md',
    'data_quality_report_latest.md',
    'data_quality_sources_latest.csv',
    'data_quality_issues_latest.csv',
    'data_quality_overlap_latest.csv',
    'data_quality_merged_summary_latest.csv',
    'data_anomaly_stocks_latest.csv',
    'data_anomaly_events_latest.csv',
    'data_cleaning_strategy_latest.md',
    'data_hygiene_inventory_latest.md',
    'data_hygiene_inventory_latest.json',
    'data_hygiene_cleanup_plan_latest.md',
    'data_hygiene_cleanup_plan_latest.json',
    'workbench_capability_inventory_latest.md',
    'workbench_capability_inventory_latest.json',
    'optimization_results_latest.json',
    'optimization_results_latest.md',
    'china_astock_strategy_comparison_latest.csv',
    'china_astock_strategy_comparison_latest.md',
    'china_astock_strategy_comparison_latest.json',
  ]) {
    copyIfExists(path.join(realReportsRoot, fileName), path.join(managedReportsRoot, fileName));
  }

  for (const fileName of ['experiment_runs.jsonl', 'experiment_runs_deleted.jsonl', 'experiment_log.md']) {
    const target = path.join(managedReportsRoot, fileName);
    if (!fs.existsSync(target)) fs.writeFileSync(target, '', 'utf-8');
  }

  process.env.QUANT_WORKFLOW_REPORTS_ROOT = managedReportsRoot;
}

seedManagedSmokeReports();

const managedWebServerEnv: Record<string, string> = Object.fromEntries(
  Object.entries(process.env).filter((entry): entry is [string, string] => typeof entry[1] === 'string'),
);
managedWebServerEnv.QUANT_WORKFLOW_REPORTS_ROOT = managedReportsRoot;

if (process.env.DSA_WEB_E2E_FAKE_ANALYSIS === '1') {
  managedWebServerEnv.JQ_USERNAME = 'managed-smoke-jq-user';
  managedWebServerEnv.JQ_PASSWORD = 'managed-smoke-jq-password';
  managedWebServerEnv.PYTHONPATH = [
    managedPythonFixtureRoot,
    managedWebServerEnv.PYTHONPATH,
  ].filter(Boolean).join(path.delimiter);
}

function resolveBackendCommand() {
  if (process.env.DSA_WEB_SMOKE_BACKEND_CMD) {
    return process.env.DSA_WEB_SMOKE_BACKEND_CMD;
  }

  const unixVenvPython = path.join(repoRoot, '.venv', 'bin', 'python');
  if (fs.existsSync(unixVenvPython)) {
    return `${unixVenvPython} main.py --webui-only --host 127.0.0.1 --port ${backendPort}`;
  }

  const windowsVenvPython = path.join(repoRoot, '.venv', 'Scripts', 'python.exe');
  if (fs.existsSync(windowsVenvPython)) {
    return `"${windowsVenvPython}" main.py --webui-only --host 127.0.0.1 --port ${backendPort}`;
  }

  return `python main.py --webui-only --host 127.0.0.1 --port ${backendPort}`;
}

export default defineConfig({
  testDir: './e2e',
  fullyParallel: false,
  retries: process.env.CI ? 2 : 0,
  reporter: 'list',
  outputDir: 'test-results/playwright-artifacts',
  use: {
    baseURL: 'http://127.0.0.1:4173',
    locale: 'zh-CN',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },
  webServer: shouldManageWebServers
    ? [
        {
          command: resolveBackendCommand(),
          cwd: repoRoot,
          env: managedWebServerEnv,
          url: `http://127.0.0.1:${backendPort}/api/v1/auth/status`,
          reuseExistingServer: false,
          timeout: 120_000,
        },
        {
          command: 'npm run dev -- --host 127.0.0.1 --port 4173',
          cwd: currentDir,
          env: managedWebServerEnv,
          url: 'http://127.0.0.1:4173',
          reuseExistingServer: false,
          timeout: 120_000,
        },
      ]
    : undefined,
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
});
