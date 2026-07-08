import { execFileSync, spawn, type ChildProcess } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { expect, test, type APIRequestContext, type Page } from '@playwright/test';

const localBaseUrl = process.env.DSA_WEB_E2E_BASE_URL || 'http://127.0.0.1:8010';
const localSmokePassword = process.env.DSA_WEB_E2E_PASSWORD || process.env.DSA_WEB_SMOKE_PASSWORD;
const e2ePrefix = `dsa_e2e_${new Date().toISOString().replace(/[-:.TZ]/g, '').slice(0, 14)}_${Math.random().toString(16).slice(2, 10).padEnd(8, '0')}`;
const currentDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(currentDir, '../../..');
const createdQuantJobIds = new Set<string>();
const createdQuantJobProcesses = new Map<string, ChildProcess>();
const createdExperimentRunIds = new Set<string>();
const createdAnalysisHistoryQueryIds = new Set<string>();
const createdChatSessionIds = new Set<string>();
const pendingSystemConfigRestores = new Map<string, string>();
const localBase = new URL(localBaseUrl);

test.use({
  baseURL: localBaseUrl,
  locale: 'zh-CN',
});

test.describe.configure({ mode: 'serial' });

type ConsoleWatch = {
  errors: string[];
  stop: () => void;
};

function assertLocalSmokeTarget() {
  const safeHosts = new Set(['127.0.0.1', 'localhost', '[::1]']);
  if (localBase.protocol !== 'http:' || !safeHosts.has(localBase.hostname)) {
    throw new Error(`Refusing to run destructive local smoke cleanup against non-local target: ${localBaseUrl}`);
  }
}

function sqlString(value: string) {
  return `'${value.replace(/'/g, "''")}'`;
}

function pad2(value: number) {
  return String(value).padStart(2, '0');
}

function formatLocalIso(value: Date) {
  const timezoneOffsetMinutes = -value.getTimezoneOffset();
  const sign = timezoneOffsetMinutes >= 0 ? '+' : '-';
  const absoluteOffsetMinutes = Math.abs(timezoneOffsetMinutes);
  return [
    `${value.getFullYear()}-${pad2(value.getMonth() + 1)}-${pad2(value.getDate())}`,
    `T${pad2(value.getHours())}:${pad2(value.getMinutes())}:${pad2(value.getSeconds())}`,
    `${sign}${pad2(Math.floor(absoluteOffsetMinutes / 60))}:${pad2(absoluteOffsetMinutes % 60)}`,
  ].join('');
}

function formatLocalCompact(value: Date) {
  return [
    value.getFullYear(),
    pad2(value.getMonth() + 1),
    pad2(value.getDate()),
    pad2(value.getHours()),
    pad2(value.getMinutes()),
    pad2(value.getSeconds()),
  ].join('');
}

function escapeRegExp(value: string) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function getDsaDatabasePath() {
  const configuredPath = process.env.DATABASE_PATH || path.join(repoRoot, 'data', 'stock_analysis.db');
  return path.isAbsolute(configuredPath)
    ? configuredPath
    : path.resolve(repoRoot, configuredPath);
}

async function waitForBackend(request: APIRequestContext) {
  const deadline = Date.now() + 30_000;
  let lastError = '';
  while (Date.now() < deadline) {
    try {
      const response = await request.get('/api/v1/health', { timeout: 5_000 });
      if (response.ok()) return;
      lastError = `HTTP ${response.status()}`;
    } catch (error) {
      lastError = error instanceof Error ? error.message : String(error);
    }
    await new Promise((resolve) => { setTimeout(resolve, 1000); });
  }
  throw new Error(`DSA local backend is not ready at ${localBaseUrl}: ${lastError}`);
}

type RawSystemConfig = {
  config_version: string;
  mask_token: string;
  items: Array<{ key: string; value: string }>;
};

type RawAlphaSiftStatus = {
  enabled?: boolean;
  available?: boolean;
  diagnostics?: Record<string, string>;
};

type RawAlphaSiftScreenTaskStatus = {
  status?: string;
  error?: string | null;
  message?: string | null;
  result?: {
    candidate_count?: number;
    candidateCount?: number;
    candidates?: Array<unknown>;
  } | null;
};

type RawAnalysisTaskStatus = {
  status?: string;
  error?: string | null;
  result?: {
    query_id?: string;
    queryId?: string;
    report?: {
      summary?: {
        analysis_summary?: string;
        analysisSummary?: string;
      };
    };
  } | null;
};

async function getSystemConfig(request: APIRequestContext): Promise<RawSystemConfig> {
  const response = await request.get('/api/v1/system/config', {
    params: { include_schema: false },
  });
  if (!response.ok()) {
    throw new Error(`Failed to read system config: HTTP ${response.status()}`);
  }
  return await response.json() as RawSystemConfig;
}

async function getSystemConfigValue(request: APIRequestContext, key: string): Promise<string> {
  const config = await getSystemConfig(request);
  return config.items.find((item) => item.key === key)?.value ?? '';
}

async function updateSystemConfigValue(request: APIRequestContext, key: string, value: string) {
  const config = await getSystemConfig(request);
  const response = await request.put('/api/v1/system/config', {
    data: {
      config_version: config.config_version,
      mask_token: config.mask_token || '******',
      reload_now: true,
      items: [{ key, value }],
    },
  });
  if (!response.ok()) {
    throw new Error(`Failed to update ${key}: HTTP ${response.status()} ${await response.text()}`);
  }
}

async function getAlphaSiftStatus(request: APIRequestContext): Promise<RawAlphaSiftStatus> {
  const response = await request.get('/api/v1/alphasift/status');
  if (!response.ok()) {
    throw new Error(`Failed to read AlphaSift status: HTTP ${response.status()}`);
  }
  return await response.json() as RawAlphaSiftStatus;
}

async function getAlphaSiftScreenTaskStatus(
  request: APIRequestContext,
  taskId: string,
): Promise<RawAlphaSiftScreenTaskStatus> {
  try {
    const response = await request.get(`/api/v1/alphasift/screen/tasks/${encodeURIComponent(taskId)}`);
    if (!response.ok()) {
      return { status: `http_${response.status()}` };
    }
    return await response.json() as RawAlphaSiftScreenTaskStatus;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return { status: `request_error:${message.slice(0, 120)}` };
  }
}

function isDeterministicFakeAnalysisEnabled() {
  return ['1', 'true', 'yes', 'on'].includes(String(
    process.env.DSA_WEB_E2E_FAKE_ANALYSIS || process.env.DSA_E2E_FAKE_ANALYSIS || '',
  ).trim().toLowerCase());
}

async function getTaskStatusForPoll(request: APIRequestContext, url: string): Promise<string> {
  try {
    const response = await request.get(url);
    if (!response.ok()) {
      return `http_${response.status()}`;
    }
    const data = await response.json() as { status?: string };
    return data.status || 'unknown';
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return `request_error:${message.slice(0, 120)}`;
  }
}

async function restorePendingSystemConfig(request: APIRequestContext) {
  for (const [key, value] of Array.from(pendingSystemConfigRestores.entries())) {
    await updateSystemConfigValue(request, key, value).catch(() => undefined);
    pendingSystemConfigRestores.delete(key);
  }
}

function watchConsole(page: Page): ConsoleWatch {
  const errors: string[] = [];
  const onConsole = (message: { type: () => string; text: () => string }) => {
    if (message.type() === 'error') {
      const text = message.text();
      if (text.includes('[StockIndexLoader] Failed to load stock index')) {
        return;
      }
      errors.push(text);
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

async function openWorkbench(page: Page, path = '/') {
  await page.addInitScript(() => {
    window.localStorage.setItem('dsa.uiLanguage', 'zh');
  });
  await page.goto(path, { waitUntil: 'domcontentloaded' });

  if (new URL(page.url()).pathname === '/login') {
    test.skip(!localSmokePassword, 'Set DSA_WEB_E2E_PASSWORD when local auth is enabled.');
    await page.locator('#password').fill(localSmokePassword!);
    await Promise.all([
      page.waitForResponse(
        (response) => response.url().includes('/api/v1/auth/login') && response.ok(),
        { timeout: 15_000 },
      ),
      page.getByRole('button', { name: /授权进入工作台|完成设置并登录/ }).click(),
    ]);
  }

  await page.waitForLoadState('domcontentloaded');
  await expect(page.locator('body')).not.toBeEmpty({ timeout: 15_000 });
}

async function expectNoHorizontalOverflow(page: Page) {
  await expect.poll(async () => page.evaluate(() => (
    document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1
  ))).toBe(true);
}

async function waitForJobToFinish(page: Page, jobId: string) {
  await expect.poll(async () => {
    return getTaskStatusForPoll(page.request, `/api/v1/quant/jobs/${encodeURIComponent(jobId)}`);
  }, {
    intervals: [1000, 1500, 2000],
    timeout: 60_000,
  }).toMatch(/^(completed|failed|cancelled)$/);
}

async function waitForAnalysisTaskToComplete(page: Page, taskId: string): Promise<RawAnalysisTaskStatus> {
  let latest: RawAnalysisTaskStatus = {};
  await expect.poll(async () => {
    const response = await page.request.get(`/api/v1/analysis/status/${encodeURIComponent(taskId)}`);
    if (!response.ok()) {
      return `http_${response.status()}`;
    }
    latest = await response.json() as RawAnalysisTaskStatus;
    return latest.status || 'unknown';
  }, {
    intervals: [500, 1000, 1500, 2000],
    timeout: 45_000,
  }).toBe('completed');
  return latest;
}

async function cleanupPortfolio(request: APIRequestContext) {
  assertLocalSmokeTarget();
  const accountsResponse = await request.get('/api/v1/portfolio/accounts', {
    params: { include_inactive: true },
  });
  if (!accountsResponse.ok()) return;
  const data = await accountsResponse.json() as { accounts?: Array<{ id: number; name?: string }> };
  for (const account of data.accounts || []) {
    if (account.name?.startsWith(e2ePrefix)) {
      const tradesResponse = await request.get('/api/v1/portfolio/trades', {
        params: { account_id: account.id, page_size: 100 },
      }).catch(() => null);
      if (tradesResponse?.ok()) {
        const trades = await tradesResponse.json() as { items?: Array<{ id: number }> };
        for (const trade of trades.items || []) {
          await request.delete(`/api/v1/portfolio/trades/${trade.id}`).catch(() => undefined);
        }
      }

      const cashResponse = await request.get('/api/v1/portfolio/cash-ledger', {
        params: { account_id: account.id, page_size: 100 },
      }).catch(() => null);
      if (cashResponse?.ok()) {
        const cashEntries = await cashResponse.json() as { items?: Array<{ id: number }> };
        for (const entry of cashEntries.items || []) {
          await request.delete(`/api/v1/portfolio/cash-ledger/${entry.id}`).catch(() => undefined);
        }
      }

      const corporateResponse = await request.get('/api/v1/portfolio/corporate-actions', {
        params: { account_id: account.id, page_size: 100 },
      }).catch(() => null);
      if (corporateResponse?.ok()) {
        const actions = await corporateResponse.json() as { items?: Array<{ id: number }> };
        for (const action of actions.items || []) {
          await request.delete(`/api/v1/portfolio/corporate-actions/${action.id}`).catch(() => undefined);
        }
      }

      await request.delete(`/api/v1/portfolio/accounts/${account.id}`).catch(() => undefined);
    }
  }
  cleanupPortfolioDatabase();
}

function cleanupPortfolioDatabase() {
  if (process.env.DSA_WEB_E2E_DB_CLEANUP === '0') return;
  assertLocalSmokeTarget();
  const databasePath = getDsaDatabasePath();
  if (!fs.existsSync(databasePath)) return;

  const accountPredicate = `name GLOB ${sqlString(`${e2ePrefix}*`)}`;
  const sql = [
    'BEGIN;',
    `DELETE FROM portfolio_daily_snapshots WHERE account_id IN (SELECT id FROM portfolio_accounts WHERE ${accountPredicate});`,
    `DELETE FROM portfolio_position_lots WHERE account_id IN (SELECT id FROM portfolio_accounts WHERE ${accountPredicate});`,
    `DELETE FROM portfolio_positions WHERE account_id IN (SELECT id FROM portfolio_accounts WHERE ${accountPredicate});`,
    `DELETE FROM portfolio_trades WHERE account_id IN (SELECT id FROM portfolio_accounts WHERE ${accountPredicate});`,
    `DELETE FROM portfolio_cash_ledger WHERE account_id IN (SELECT id FROM portfolio_accounts WHERE ${accountPredicate});`,
    `DELETE FROM portfolio_corporate_actions WHERE account_id IN (SELECT id FROM portfolio_accounts WHERE ${accountPredicate});`,
    `DELETE FROM portfolio_accounts WHERE ${accountPredicate};`,
    'COMMIT;',
  ].join(' ');

  execFileSync('sqlite3', [databasePath, sql], { stdio: 'pipe' });
}

function cleanupAnalysisHistoryDatabase(queryIds = Array.from(createdAnalysisHistoryQueryIds)) {
  if (process.env.DSA_WEB_E2E_DB_CLEANUP === '0') return;
  assertLocalSmokeTarget();
  const safeQueryIds = queryIds.filter((queryId) => queryId.startsWith('dsa_e2e_'));
  if (safeQueryIds.length === 0) return;
  const databasePath = getDsaDatabasePath();
  if (!fs.existsSync(databasePath)) return;

  const quotedIds = safeQueryIds.map(sqlString).join(',');
  execFileSync('sqlite3', [databasePath, `DELETE FROM analysis_history WHERE query_id IN (${quotedIds});`], { stdio: 'pipe' });
  for (const queryId of safeQueryIds) {
    createdAnalysisHistoryQueryIds.delete(queryId);
  }
}

function cleanupE2eFakeAnalysisHistoryDatabase() {
  if (process.env.DSA_WEB_E2E_DB_CLEANUP === '0') return;
  assertLocalSmokeTarget();
  const databasePath = getDsaDatabasePath();
  if (!fs.existsSync(databasePath)) return;

  const marker = '%dsa_e2e_fake_analysis%';
  const sql = [
    'DELETE FROM analysis_history',
    `WHERE query_id GLOB 'dsa_e2e_fake_analysis_*'`,
    `OR raw_result LIKE ${sqlString(marker)}`,
    `OR analysis_summary LIKE ${sqlString(marker)};`,
  ].join(' ');
  execFileSync('sqlite3', [databasePath, sql], { stdio: 'pipe' });
}

function createTemporaryAnalysisHistory() {
  assertLocalSmokeTarget();
  const databasePath = getDsaDatabasePath();
  if (!fs.existsSync(databasePath)) {
    throw new Error(`DSA database not found: ${databasePath}`);
  }

  const queryId = `${e2ePrefix}_history`;
  const stockCode = '689997';
  const stockName = 'E2E历史';
  const summary = `${e2ePrefix} 历史报告摘要`;
  const now = new Date().toISOString();
  cleanupAnalysisHistoryDatabase([queryId]);

  const rawResult = {
    code: stockCode,
    name: stockName,
    report_language: 'zh',
    report_type: 'detailed',
    sentiment_score: 67,
    operation_advice: '持有观察',
    trend_prediction: '震荡上行',
    analysis_summary: summary,
    action: 'hold',
    action_label: '持有',
    current_price: 12.34,
    change_pct: 1.23,
    technical_analysis: `${e2ePrefix} 技术面烟测内容`,
    ma_analysis: '均线维持多头排列。',
    volume_analysis: '量能温和放大。',
    news_summary: '暂无重大负面消息。',
    risk_warning: '仅用于本地 smoke 测试。',
    key_points: '历史详情、Markdown 抽屉和删除链路。',
    model_used: 'dsa-e2e',
    success: true,
  };
  const contextSnapshot = {
    report_language: 'zh',
    realtime_quote: {
      current_price: 12.34,
      change_pct: 1.23,
    },
  };
  const sql = `
    INSERT INTO analysis_history (
      query_id, code, name, report_type, sentiment_score, operation_advice,
      trend_prediction, analysis_summary, raw_result, news_content,
      context_snapshot, ideal_buy, secondary_buy, stop_loss, take_profit,
      created_at
    ) VALUES (
      ${sqlString(queryId)},
      ${sqlString(stockCode)},
      ${sqlString(stockName)},
      'detailed',
      67,
      '持有观察',
      '震荡上行',
      ${sqlString(summary)},
      ${sqlString(JSON.stringify(rawResult))},
      '本地 smoke 临时新闻摘要',
      ${sqlString(JSON.stringify(contextSnapshot))},
      11.8,
      11.2,
      10.5,
      14.8,
      ${sqlString(now)}
    );
    SELECT id FROM analysis_history WHERE query_id = ${sqlString(queryId)} ORDER BY id DESC LIMIT 1;
  `;
  const output = execFileSync('sqlite3', [databasePath, sql], { encoding: 'utf-8', stdio: 'pipe' }).trim();
  const recordId = Number(output.split(/\r?\n/).pop());
  if (!Number.isFinite(recordId)) {
    throw new Error(`Failed to create temporary analysis history row for ${queryId}`);
  }
  createdAnalysisHistoryQueryIds.add(queryId);
  return { queryId, recordId, stockCode, stockName, summary };
}

async function cleanupAlerts(request: APIRequestContext) {
  assertLocalSmokeTarget();
  const rulesResponse = await request.get('/api/v1/alerts/rules', { params: { page_size: 100 } });
  if (!rulesResponse.ok()) return;
  const data = await rulesResponse.json() as { items?: Array<{ id: number; name?: string }> };
  for (const rule of data.items || []) {
    if (rule.name?.startsWith(e2ePrefix)) {
      await request.delete(`/api/v1/alerts/rules/${rule.id}`).catch(() => undefined);
    }
  }
}

async function cleanupChatSessions(request: APIRequestContext) {
  assertLocalSmokeTarget();
  const sessionsResponse = await request.get('/api/v1/agent/chat/sessions', { params: { limit: 100 } });
  if (!sessionsResponse.ok()) return;
  const data = await sessionsResponse.json() as { sessions?: Array<{ session_id: string; title?: string }> };
  for (const session of data.sessions || []) {
    if (
      session.title?.includes(e2ePrefix)
      || session.session_id.includes(e2ePrefix)
    ) {
      await request.delete(`/api/v1/agent/chat/sessions/${encodeURIComponent(session.session_id)}`).catch(() => undefined);
    }
  }
  cleanupChatSessionDatabase();
}

function cleanupChatSessionDatabase(sessionIds = Array.from(createdChatSessionIds)) {
  if (process.env.DSA_WEB_E2E_DB_CLEANUP === '0') return;
  assertLocalSmokeTarget();
  const databasePath = getDsaDatabasePath();
  if (!fs.existsSync(databasePath)) return;

  const exactSessionIds = sessionIds.filter((sessionId) => (
    sessionId.startsWith('dsa_e2e_') || /^[0-9a-f-]{32,36}$/i.test(sessionId)
  ));
  const exactPredicate = exactSessionIds.length > 0
    ? ` OR session_id IN (${exactSessionIds.map(sqlString).join(',')})`
    : '';
  const sessionSelect = [
    'SELECT session_id FROM conversation_messages',
    `WHERE content LIKE ${sqlString(`%${e2ePrefix}%`)}${exactPredicate}`,
  ].join(' ');
  const sql = [
    'BEGIN;',
    `DELETE FROM agent_provider_turns WHERE session_id IN (${sessionSelect});`,
    `DELETE FROM conversation_summaries WHERE session_id IN (${sessionSelect});`,
    `DELETE FROM conversation_messages WHERE session_id IN (${sessionSelect});`,
    'COMMIT;',
  ].join(' ');
  execFileSync('sqlite3', [databasePath, sql], { stdio: 'pipe' });
  for (const sessionId of exactSessionIds) {
    createdChatSessionIds.delete(sessionId);
  }
}

function createTemporaryChatSession() {
  assertLocalSmokeTarget();
  const databasePath = getDsaDatabasePath();
  if (!fs.existsSync(databasePath)) {
    throw new Error(`DSA database not found: ${databasePath}`);
  }

  const sessionId = `${e2ePrefix}_chat_session`;
  const userContent = `${e2ePrefix}_chat_lifecycle_probe 600519`;
  const assistantContent = `# ${e2ePrefix} 问股回复\n\n这是本地 smoke 测试会话，用于验证 Markdown 导出和永久删除。`;
  const now = new Date().toISOString();
  cleanupChatSessionDatabase([sessionId]);

  const sql = [
    'INSERT INTO conversation_messages (session_id, role, content, created_at)',
    `VALUES (${sqlString(sessionId)}, 'user', ${sqlString(userContent)}, ${sqlString(now)});`,
    'INSERT INTO conversation_messages (session_id, role, content, created_at)',
    `VALUES (${sqlString(sessionId)}, 'assistant', ${sqlString(assistantContent)}, ${sqlString(now)});`,
  ].join(' ');
  execFileSync('sqlite3', [databasePath, sql], { stdio: 'pipe' });
  createdChatSessionIds.add(sessionId);
  return { sessionId, userContent, assistantContent };
}

function getQuantJobsRoot() {
  return path.join(getQuantReportsRoot(), 'jobs');
}

function getQuantReportsRoot() {
  const configuredReportsRoot = process.env.QUANT_WORKFLOW_REPORTS_ROOT;
  if (configuredReportsRoot) {
    return path.isAbsolute(configuredReportsRoot)
      ? configuredReportsRoot
      : path.resolve(repoRoot, configuredReportsRoot);
  }
  const configuredRoot = process.env.QUANT_WORKFLOW_ROOT || path.resolve(repoRoot, 'quant_workflow');
  const quantWorkflowRoot = path.isAbsolute(configuredRoot)
    ? configuredRoot
    : path.resolve(repoRoot, configuredRoot);
  return path.join(quantWorkflowRoot, 'reports');
}

function isSafeQuantJobId(jobId: string) {
  return /^qj_\d{14}_[0-9a-f]{8}$/.test(jobId);
}

function cleanupQuantJobDatabase(jobIds: string[]) {
  if (process.env.DSA_WEB_E2E_QUANT_JOB_CLEANUP === '0') return;
  assertLocalSmokeTarget();
  const safeJobIds = jobIds.filter(isSafeQuantJobId);
  if (safeJobIds.length === 0) return;

  const jobsRoot = getQuantJobsRoot();
  const databasePath = path.join(jobsRoot, 'index.sqlite');
  if (fs.existsSync(databasePath)) {
    const quotedIds = safeJobIds.map((jobId) => `'${jobId}'`).join(',');
    execFileSync('sqlite3', [databasePath, `DELETE FROM jobs WHERE job_id IN (${quotedIds});`], { stdio: 'pipe' });
  }

  for (const jobId of safeJobIds) {
    fs.rmSync(path.join(jobsRoot, jobId), { recursive: true, force: true });
    createdQuantJobIds.delete(jobId);
    const child = createdQuantJobProcesses.get(jobId);
    if (child?.pid && isPidAlive(child.pid)) {
      terminateProcessGroup(child.pid);
    }
    createdQuantJobProcesses.delete(jobId);
  }
}

function isPidAlive(pid: number) {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

function terminateProcessGroup(pid: number) {
  for (const signal of ['SIGTERM', 'SIGKILL'] as const) {
    try {
      process.kill(-pid, signal);
    } catch {
      try {
        process.kill(pid, signal);
      } catch {
        // Already exited.
      }
    }
    if (!isPidAlive(pid)) return;
  }
}

function createTemporaryQuantArtifactJob() {
  assertLocalSmokeTarget();
  const timestamp = new Date();
  const timestampCompact = formatLocalCompact(timestamp);
  const jobId = `qj_${timestampCompact}_${Math.random().toString(16).slice(2, 10).padEnd(8, '0')}`;
  const jobsRoot = getQuantJobsRoot();
  const jobDir = path.join(jobsRoot, jobId);
  const logPath = path.join(jobDir, 'job.log');
  const artifactPath = path.join(jobDir, `${e2ePrefix}_artifact_report.md`);
  const databasePath = path.join(jobsRoot, 'index.sqlite');
  const now = formatLocalIso(timestamp);

  fs.mkdirSync(jobDir, { recursive: true });
  fs.writeFileSync(logPath, `dsa e2e artifact job ${jobId}\n`, 'utf-8');
  fs.writeFileSync(
    artifactPath,
    `# DSA E2E Artifact\n\nRun: ${jobId}\n\nMarker: ${e2ePrefix}_artifact_marker\n`,
    'utf-8',
  );

  const sql = [
    `CREATE TABLE IF NOT EXISTS jobs (
      job_id TEXT PRIMARY KEY,
      job_type TEXT NOT NULL,
      label TEXT NOT NULL,
      status TEXT NOT NULL,
      params_json TEXT NOT NULL,
      command_json TEXT NOT NULL,
      cwd TEXT NOT NULL,
      pid INTEGER,
      return_code INTEGER,
      created_at TEXT NOT NULL,
      started_at TEXT,
      completed_at TEXT,
      updated_at TEXT NOT NULL,
      error TEXT,
      log_path TEXT NOT NULL,
      artifact_paths_json TEXT NOT NULL
    );`,
    `INSERT OR REPLACE INTO jobs (
      job_id, job_type, label, status, params_json, command_json, cwd,
      pid, return_code, created_at, started_at, completed_at, updated_at,
      error, log_path, artifact_paths_json
    ) VALUES (
      ${sqlString(jobId)},
      'report_data_quality',
      '生成数据质量报告',
      'completed',
      '{}',
      '[]',
      ${sqlString(repoRoot)},
      NULL,
      0,
      ${sqlString(now)},
      ${sqlString(now)},
      ${sqlString(now)},
      ${sqlString(now)},
      NULL,
      ${sqlString(logPath)},
      ${sqlString(JSON.stringify([artifactPath]))}
    );`,
  ].join(' ');

  execFileSync('sqlite3', [databasePath, sql], { stdio: 'pipe' });
  createdQuantJobIds.add(jobId);
  return { jobId, artifactPath };
}

function createTemporaryRunningQuantJob() {
  assertLocalSmokeTarget();
  const timestamp = new Date();
  const timestampCompact = formatLocalCompact(timestamp);
  const jobId = `qj_${timestampCompact}_${Math.random().toString(16).slice(2, 10).padEnd(8, '0')}`;
  const jobsRoot = getQuantJobsRoot();
  const jobDir = path.join(jobsRoot, jobId);
  const logPath = path.join(jobDir, 'job.log');
  const databasePath = path.join(jobsRoot, 'index.sqlite');
  const now = formatLocalIso(timestamp);

  fs.mkdirSync(jobDir, { recursive: true });
  fs.writeFileSync(logPath, `dsa e2e cancellable job ${jobId}\n`, 'utf-8');
  const child = spawn(
    process.execPath,
    ['-e', 'setTimeout(() => {}, 120000)'],
    { detached: true, stdio: 'ignore' },
  );
  child.unref();
  if (!child.pid) {
    throw new Error(`Failed to start cancellable quant test process for ${jobId}`);
  }

  const sql = [
    `CREATE TABLE IF NOT EXISTS jobs (
      job_id TEXT PRIMARY KEY,
      job_type TEXT NOT NULL,
      label TEXT NOT NULL,
      status TEXT NOT NULL,
      params_json TEXT NOT NULL,
      command_json TEXT NOT NULL,
      cwd TEXT NOT NULL,
      pid INTEGER,
      return_code INTEGER,
      created_at TEXT NOT NULL,
      started_at TEXT,
      completed_at TEXT,
      updated_at TEXT NOT NULL,
      error TEXT,
      log_path TEXT NOT NULL,
      artifact_paths_json TEXT NOT NULL
    );`,
    `INSERT OR REPLACE INTO jobs (
      job_id, job_type, label, status, params_json, command_json, cwd,
      pid, return_code, created_at, started_at, completed_at, updated_at,
      error, log_path, artifact_paths_json
    ) VALUES (
      ${sqlString(jobId)},
      'check_qlib_data',
      '检查 Qlib 数据',
      'running',
      '{}',
      ${sqlString(JSON.stringify([process.execPath, '-e', 'setTimeout(() => {}, 120000)']))},
      ${sqlString(repoRoot)},
      ${child.pid},
      NULL,
      ${sqlString(now)},
      ${sqlString(now)},
      NULL,
      ${sqlString(now)},
      NULL,
      ${sqlString(logPath)},
      '[]'
    );`,
  ].join(' ');

  execFileSync('sqlite3', [databasePath, sql], { stdio: 'pipe' });
  createdQuantJobIds.add(jobId);
  createdQuantJobProcesses.set(jobId, child);
  return { jobId, pid: child.pid };
}

async function cleanupQuantJobs(request: APIRequestContext) {
  const jobIds = Array.from(createdQuantJobIds);
  for (const jobId of jobIds) {
    const response = await request.get(`/api/v1/quant/jobs/${encodeURIComponent(jobId)}`).catch(() => null);
    if (!response?.ok()) continue;
    const data = await response.json() as { status?: string };
    if (data.status === 'pending' || data.status === 'running') {
      await request.post(`/api/v1/quant/jobs/${encodeURIComponent(jobId)}/cancel`).catch(() => undefined);
    }
  }
  cleanupQuantJobDatabase(jobIds);
}

function stripExperimentRunFromFile(filePath: string, runIds: string[]) {
  if (!fs.existsSync(filePath)) return;
  const original = fs.readFileSync(filePath, 'utf-8');
  const filtered = original
    .split(/\r?\n/)
    .filter((line) => !runIds.some((runId) => line.includes(runId)))
    .join('\n')
    .replace(/\n{3,}/g, '\n\n')
    .trimEnd();
  fs.writeFileSync(filePath, filtered ? `${filtered}\n` : '', 'utf-8');
}

function stripExperimentNoteBlocks(filePath: string, runIds: string[]) {
  if (!fs.existsSync(filePath)) return;
  let content = fs.readFileSync(filePath, 'utf-8');
  for (const runId of runIds) {
    const escaped = runId.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    content = content.replace(
      new RegExp(`\\n?<!-- experiment-run:${escaped} -->[\\s\\S]*?<!-- /experiment-run:${escaped} -->\\n?`, 'g'),
      '\n',
    );
  }
  fs.writeFileSync(filePath, content.replace(/\n{3,}/g, '\n\n').trimEnd() + (content.trim() ? '\n' : ''), 'utf-8');
}

function cleanupExperimentRuns(runIds = Array.from(createdExperimentRunIds)) {
  const safeRunIds = runIds.filter((runId) => runId.startsWith('dsa_e2e_'));
  if (safeRunIds.length === 0) return;
  const reportsRoot = getQuantReportsRoot();
  stripExperimentRunFromFile(path.join(reportsRoot, 'experiment_runs.jsonl'), safeRunIds);
  stripExperimentRunFromFile(path.join(reportsRoot, 'experiment_runs_deleted.jsonl'), safeRunIds);
  stripExperimentNoteBlocks(path.join(reportsRoot, 'experiment_log.md'), safeRunIds);
  for (const runId of safeRunIds) {
    createdExperimentRunIds.delete(runId);
  }
}

function createTemporaryExperimentRun(runId: string, withMarkedNote: boolean) {
  const reportsRoot = getQuantReportsRoot();
  fs.mkdirSync(reportsRoot, { recursive: true });
  cleanupExperimentRuns([runId]);
  createdExperimentRunIds.add(runId);

  const run = {
    run_id: runId,
    experiment_name: `${runId}_baseline_smoke`,
    status: 'completed',
    model: 'CodexSmokeModel',
    strategy: 'CodexSmokeStrategy',
    dataset_range: '2026-07-01 ~ 2026-07-02',
    conclusion: 'dsa e2e local smoke conclusion',
    next_step: 'dsa e2e cleanup',
    metrics: {
      '1day.excess_return_with_cost.annualized_return': 0.001,
    },
  };
  fs.appendFileSync(
    path.join(reportsRoot, 'experiment_runs.jsonl'),
    `${JSON.stringify(run)}\n`,
    'utf-8',
  );

  if (withMarkedNote) {
    fs.appendFileSync(
      path.join(reportsRoot, 'experiment_log.md'),
      `\n<!-- experiment-run:${runId} -->\n## ${runId} local smoke note\n\n- 结论：dsa e2e local smoke note。\n<!-- /experiment-run:${runId} -->\n`,
      'utf-8',
    );
  }
}

test.afterEach(async ({ request }) => {
  await restorePendingSystemConfig(request);
  await cleanupQuantJobs(request);
  cleanupExperimentRuns();
  cleanupAnalysisHistoryDatabase();
  cleanupE2eFakeAnalysisHistoryDatabase();
  await cleanupAlerts(request);
  await cleanupPortfolio(request);
  await cleanupChatSessions(request);
});

test.beforeAll(async ({ request }) => {
  assertLocalSmokeTarget();
  await waitForBackend(request);
});

test('main workbench routes render in Chinese without runtime errors or horizontal overflow', async ({ page }) => {
  const consoleWatch = watchConsole(page);
  const routes = [
    { path: '/', text: '个股栏' },
    { path: '/research', text: '研究工作台' },
    { path: '/workflow', text: '量化任务中心' },
    { path: '/backtest', text: 'Qlib模型/策略回测' },
    { path: '/chat', text: '问股' },
    { path: '/portfolio', text: '持仓管理' },
    { path: '/alerts', text: '告警中心' },
    { path: '/settings', text: '系统设置' },
    { path: '/screening', text: 'AlphaSift 选股' },
  ];

  try {
    for (const route of routes) {
      await openWorkbench(page, route.path);
      await expect(page.locator('body')).toContainText(route.text, { timeout: 15_000 });
      await expectNoHorizontalOverflow(page);
    }
    expect(consoleWatch.errors).toEqual([]);
  } finally {
    consoleWatch.stop();
  }
});

test('research page shows capability acceptance matrix and deferred external-tool boundaries', async ({ page }) => {
  const consoleWatch = watchConsole(page);
  try {
    await openWorkbench(page, '/research');
    await expect(page.getByRole('heading', { name: '研究工作台' })).toBeVisible({ timeout: 15_000 });
    await expect(page.getByRole('heading', { name: '能力验收矩阵' })).toBeVisible();
    await expect(page.getByRole('heading', { name: '功能入口盘点' })).toBeVisible();
    await expect(page.getByRole('heading', { name: '后续开发队列' })).toBeVisible();
    await expect(page.getByText('模型质量和清仓信号')).toBeVisible();
    await expect(page.getByText('场内基金/ETF深度版')).toBeVisible();
    await expect(page.getByText('daily_stock_analysis WebUI').last()).toBeVisible();
    await expect(page.getByText(/quant_workflow/).last()).toBeVisible();
    await expect(page.getByText(/Qlib/).last()).toBeVisible();
    await expect(page.getByText('china-astock-quant').last()).toBeVisible();
    await expect(page.getByText('同花顺桌面端').last()).toBeVisible();
    await expect(page.getByText('交易执行').last()).toBeVisible();
    await expect(page.getByText(/已检测到本机同花顺|未在默认路径检测到同花顺/).last()).toBeVisible();
    await expect(page.getByText(/不读取账号密码/).last()).toBeVisible();
    await expect(page.getByText(/不读取券商密码/).last()).toBeVisible();
    await expect(page.getByText(/不自动下单/).last()).toBeVisible();
    await expect(page.getByText('测试脚本').last()).toBeVisible();
    await expect(page.getByText('浏览器执行').last()).toBeVisible();
    await expect(page.getByText('opencode复审').last()).toBeVisible();
    await expectNoHorizontalOverflow(page);
    expect(consoleWatch.errors).toEqual([]);
  } finally {
    consoleWatch.stop();
  }
});

test('workflow page can launch the safe Qlib data check and show its log', async ({ page }) => {
  const consoleWatch = watchConsole(page);
  try {
    await openWorkbench(page, '/workflow');
    await expect(page.getByRole('heading', { name: '量化任务中心' })).toBeVisible();

    const responsePromise = page.waitForResponse(
      (response) => response.url().includes('/api/v1/quant/jobs') && response.request().method() === 'POST',
      { timeout: 15_000 },
    );
    await page.getByRole('button', { name: '运行 检查 Qlib 数据' }).click();
    const submitResponse = await responsePromise;
    expect(submitResponse.ok()).toBeTruthy();
    const job = await submitResponse.json() as { job_id: string };
    createdQuantJobIds.add(job.job_id);

    await waitForJobToFinish(page, job.job_id);
    await page.reload({ waitUntil: 'domcontentloaded' });
    await expect(page.getByText(job.job_id)).toBeVisible({ timeout: 15_000 });
    await page.getByText(job.job_id).click();
    await expect(page.locator('pre')).toContainText(/Qlib|provider|calendar|instrument|qlib/i, { timeout: 15_000 });

    expect(consoleWatch.errors).toEqual([]);
  } finally {
    consoleWatch.stop();
  }
});

test('workflow page can launch the daily guide cleaning preflight and preview the plan', async ({ page }) => {
  const consoleWatch = watchConsole(page);
  try {
    await openWorkbench(page, '/workflow');
    await expect(page.getByRole('heading', { name: '量化任务中心' })).toBeVisible();
    await expect(page.getByText('按顺序完成检查、报告、预演、训练和记录。')).toBeVisible({ timeout: 15_000 });
    await expect(page.getByRole('button', { name: '今日流程 清洗预演 预演' })).toBeVisible();
    await expect(page.getByRole('heading', { name: '应用清洗策略' })).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText('根据异常清单生成 cleaned Qlib provider，不覆盖原始 canonical provider。')).toBeVisible();
    await expect(page.getByLabel('应用清洗策略 只预演不写入')).toBeVisible();
    await expect(page.getByLabel('应用清洗策略 不覆盖已有文件')).toBeVisible();
    await expect(page.getByLabel('应用清洗策略 仅生成 cleaned Parquet')).toBeVisible();

    const responsePromise = page.waitForResponse(
      (response) => response.url().includes('/api/v1/quant/jobs') && response.request().method() === 'POST',
      { timeout: 15_000 },
    );
    await page.getByRole('button', { name: '今日流程 清洗预演 预演' }).click();
    const submitResponse = await responsePromise;
    expect(submitResponse.ok()).toBeTruthy();
    expect(submitResponse.request().postDataJSON()).toMatchObject({
      job_type: 'apply_data_cleaning',
      params: { dry_run: true, no_overwrite: true },
    });
    const job = await submitResponse.json() as { job_id: string };
    createdQuantJobIds.add(job.job_id);

    await waitForJobToFinish(page, job.job_id);
    await page.reload({ waitUntil: 'domcontentloaded' });
    await expect(page.getByText(job.job_id)).toBeVisible({ timeout: 15_000 });
    const previewButton = page.getByRole('button', { name: /预览 data_cleaning_applied_latest\.md/ });
    await expect(previewButton).toBeVisible({ timeout: 15_000 });
    await previewButton.click();
    await expect(page.locator('pre')).toContainText('Data Cleaning Dry Run', { timeout: 15_000 });
    await expect(page.locator('pre')).toContainText('no cleaned parquet or Qlib files were written', { timeout: 15_000 });

    const statusResponse = await page.request.get('/api/v1/quant/status');
    expect(statusResponse.ok()).toBeTruthy();
    const status = await statusResponse.json() as {
      available_job_types?: Record<string, { allowed_params?: string[] }>;
    };
    expect(status.available_job_types?.apply_data_cleaning?.allowed_params).toEqual([
      'dry_run',
      'no_overwrite',
      'skip_qlib_convert',
    ]);
    expect(consoleWatch.errors).toEqual([]);
  } finally {
    consoleWatch.stop();
  }
});

test('workflow page can launch the Baostock refresh plan from the browser', async ({ page }) => {
  const consoleWatch = watchConsole(page);
  try {
    await openWorkbench(page, '/workflow');
    await expect(page.getByRole('heading', { name: '量化任务中心' })).toBeVisible();
    await expect(page.getByRole('heading', { name: '准备 Baostock 增量清单' })).toBeVisible({ timeout: 15_000 });

    await page.getByLabel('准备 Baostock 增量清单 股票数量上限').fill('3');

    const responsePromise = page.waitForResponse(
      (response) => response.url().includes('/api/v1/quant/jobs') && response.request().method() === 'POST',
      { timeout: 15_000 },
    );
    await page.getByRole('button', { name: '运行 准备 Baostock 增量清单' }).click();
    const submitResponse = await responsePromise;
    expect(submitResponse.ok()).toBeTruthy();
    expect(submitResponse.request().postDataJSON()).toMatchObject({
      job_type: 'prepare_baostock_refresh',
      params: { limit: '3' },
    });
    const job = await submitResponse.json() as { job_id: string };
    createdQuantJobIds.add(job.job_id);

    await waitForJobToFinish(page, job.job_id);
    await page.reload({ waitUntil: 'domcontentloaded' });
    await expect(page.getByText(job.job_id)).toBeVisible({ timeout: 15_000 });
    const previewButton = page.getByRole('button', { name: /预览 baostock_refresh_plan_latest\.md/ });
    await expect(previewButton).toBeVisible({ timeout: 15_000 });
    await previewButton.click();
    await expect(page.locator('pre')).toContainText('Baostock Refresh Plan', { timeout: 15_000 });
    await expect(page.locator('pre')).toContainText('Refresh source: Baostock daily qfq bars', { timeout: 15_000 });

    expect(consoleWatch.errors).toEqual([]);
  } finally {
    consoleWatch.stop();
  }
});

test('workflow page can launch a JQData connection dry run from the browser without writing parquet', async ({ page }) => {
  const consoleWatch = watchConsole(page);
  try {
    await openWorkbench(page, '/workflow');
    await expect(page.getByRole('heading', { name: '量化任务中心' })).toBeVisible();
    await expect(page.getByRole('heading', { name: '验证 JQData 连接' })).toBeVisible({ timeout: 15_000 });

    await page.getByLabel('验证 JQData 连接 股票池').fill('csi300');
    await page.getByLabel('验证 JQData 连接 验证样本数').fill('1');
    await page.getByLabel('验证 JQData 连接 请求间隔秒').fill('0');

    const responsePromise = page.waitForResponse(
      (response) => response.url().includes('/api/v1/quant/jobs') && response.request().method() === 'POST',
      { timeout: 15_000 },
    );
    await page.getByRole('button', { name: '运行 验证 JQData 连接' }).click();
    const submitResponse = await responsePromise;
    expect(submitResponse.ok()).toBeTruthy();
    expect(submitResponse.request().postDataJSON()).toMatchObject({
      job_type: 'download_jqdata',
      params: { universe: 'csi300', limit: '1', sleep: '0', dry_run: true },
    });
    const job = await submitResponse.json() as { job_id: string };
    createdQuantJobIds.add(job.job_id);

    await waitForJobToFinish(page, job.job_id);
    await page.reload({ waitUntil: 'domcontentloaded' });
    await expect(page.getByText(job.job_id)).toBeVisible({ timeout: 15_000 });
    await page.getByText(job.job_id).click();
    await expect(page.locator('pre')).toContainText('JQData dry run OK', { timeout: 15_000 });
    await expect(page.locator('pre')).toContainText('No parquet files were written', { timeout: 15_000 });

    expect(consoleWatch.errors).toEqual([]);
  } finally {
    consoleWatch.stop();
  }
});

test('workflow page can launch a Qlib provider repair dry run from the browser', async ({ page }) => {
  const consoleWatch = watchConsole(page);
  try {
    await openWorkbench(page, '/workflow');
    await expect(page.getByRole('heading', { name: '量化任务中心' })).toBeVisible();
    await expect(page.getByRole('heading', { name: '预演修复 Qlib Provider' })).toBeVisible({ timeout: 15_000 });
    await expect(page.getByLabel('修复 Qlib Provider 只检查不写入')).toHaveCount(0);

    const responsePromise = page.waitForResponse(
      (response) => response.url().includes('/api/v1/quant/jobs') && response.request().method() === 'POST',
      { timeout: 15_000 },
    );
    await page.getByRole('button', { name: '运行 预演修复 Qlib Provider' }).click();
    const submitResponse = await responsePromise;
    expect(submitResponse.ok()).toBeTruthy();
    expect(submitResponse.request().postDataJSON()).toMatchObject({
      job_type: 'repair_qlib_provider',
      params: {},
    });
    const job = await submitResponse.json() as { job_id: string };
    createdQuantJobIds.add(job.job_id);

    await waitForJobToFinish(page, job.job_id);
    await page.reload({ waitUntil: 'domcontentloaded' });
    await expect(page.getByText(job.job_id)).toBeVisible({ timeout: 15_000 });
    await page.getByText(job.job_id).click();
    await expect(page.locator('pre')).toContainText('Feature directory renames', { timeout: 15_000 });
    await expect(page.locator('pre')).toContainText('csi300.txt', { timeout: 15_000 });

    expect(consoleWatch.errors).toEqual([]);
  } finally {
    consoleWatch.stop();
  }
});

test('workflow page can launch provider maintenance dry-runs from the browser', async ({ page }) => {
  const consoleWatch = watchConsole(page);
  try {
    await openWorkbench(page, '/workflow');
    await expect(page.getByRole('heading', { name: '量化任务中心' })).toBeVisible();

    const csiResponsePromise = page.waitForResponse(
      (response) => response.url().includes('/api/v1/quant/jobs') && response.request().method() === 'POST',
      { timeout: 15_000 },
    );
    await page.getByRole('button', { name: '运行 预演生成 CSI300 股票池' }).click();
    const csiResponse = await csiResponsePromise;
    expect(csiResponse.ok()).toBeTruthy();
    expect(csiResponse.request().postDataJSON()).toMatchObject({
      job_type: 'generate_csi300',
      params: {},
    });
    const csiJob = await csiResponse.json() as { job_id: string };
    createdQuantJobIds.add(csiJob.job_id);

    await waitForJobToFinish(page, csiJob.job_id);
    await page.reload({ waitUntil: 'domcontentloaded' });
    await expect(page.getByText(csiJob.job_id)).toBeVisible({ timeout: 15_000 });
    await page.getByText(csiJob.job_id).click();
    await expect(page.locator('pre')).toContainText('Dry run OK', { timeout: 15_000 });
    await expect(page.locator('pre')).toContainText('CSI300 constituents', { timeout: 15_000 });

    await page.reload({ waitUntil: 'domcontentloaded' });
    const benchmarkResponsePromise = page.waitForResponse(
      (response) => response.url().includes('/api/v1/quant/jobs') && response.request().method() === 'POST',
      { timeout: 15_000 },
    );
    await page.getByRole('button', { name: '运行 预演添加沪深300 benchmark' }).click();
    const benchmarkResponse = await benchmarkResponsePromise;
    expect(benchmarkResponse.ok()).toBeTruthy();
    expect(benchmarkResponse.request().postDataJSON()).toMatchObject({
      job_type: 'add_index_benchmark',
      params: {},
    });
    const benchmarkJob = await benchmarkResponse.json() as { job_id: string };
    createdQuantJobIds.add(benchmarkJob.job_id);

    await waitForJobToFinish(page, benchmarkJob.job_id);
    await page.reload({ waitUntil: 'domcontentloaded' });
    await expect(page.getByText(benchmarkJob.job_id)).toBeVisible({ timeout: 15_000 });
    await page.getByText(benchmarkJob.job_id).click();
    await expect(page.locator('pre')).toContainText('Dry run OK', { timeout: 15_000 });
    await expect(page.locator('pre')).toContainText('benchmark', { timeout: 15_000 });

    expect(consoleWatch.errors).toEqual([]);
  } finally {
    consoleWatch.stop();
  }
});

test('workflow page can launch an isolated Qlib merge smoke fixture from the browser', async ({ page }) => {
  const consoleWatch = watchConsole(page);
  try {
    await openWorkbench(page, '/workflow');
    await expect(page.getByRole('heading', { name: '量化任务中心' })).toBeVisible();
    await expect(page.getByRole('heading', { name: '合并并转换为 Qlib' })).toBeVisible({ timeout: 15_000 });

    await page.getByLabel('合并并转换为 Qlib 使用隔离小样本演练').check();
    await page.getByLabel('合并并转换为 Qlib 跳过已有输出文件').check();

    const responsePromise = page.waitForResponse(
      (response) => response.url().includes('/api/v1/quant/jobs') && response.request().method() === 'POST',
      { timeout: 15_000 },
    );
    await page.getByRole('button', { name: '运行 合并并转换为 Qlib' }).click();
    const submitResponse = await responsePromise;
    expect(submitResponse.ok()).toBeTruthy();
    expect(submitResponse.request().postDataJSON()).toMatchObject({
      job_type: 'merge_to_qlib',
      params: { smoke: true, no_overwrite: true },
    });
    const job = await submitResponse.json() as { job_id: string };
    createdQuantJobIds.add(job.job_id);

    await waitForJobToFinish(page, job.job_id);
    await page.reload({ waitUntil: 'domcontentloaded' });
    await expect(page.getByText(job.job_id)).toBeVisible({ timeout: 15_000 });
    await page.getByText(job.job_id).click();
    await expect(page.locator('pre')).toContainText('Merge fixture root', { timeout: 15_000 });
    await expect(page.locator('pre')).toContainText('Qlib bin: 1 stocks, 3 days', { timeout: 15_000 });

    expect(consoleWatch.errors).toEqual([]);
  } finally {
    consoleWatch.stop();
  }
});

test('workflow page can launch a sampled data quality report from the browser', async ({ page }) => {
  const consoleWatch = watchConsole(page);
  try {
    await openWorkbench(page, '/workflow');
    await expect(page.getByRole('heading', { name: '量化任务中心' })).toBeVisible();
    await expect(page.getByRole('heading', { name: '生成数据质量报告' })).toBeVisible({ timeout: 15_000 });

    await page.getByLabel('生成数据质量报告 重叠样本数').fill('5');
    await page.getByLabel('生成数据质量报告 合并样本数').fill('5');

    const responsePromise = page.waitForResponse(
      (response) => response.url().includes('/api/v1/quant/jobs') && response.request().method() === 'POST',
      { timeout: 15_000 },
    );
    await page.getByRole('button', { name: '运行 生成数据质量报告' }).click();
    const submitResponse = await responsePromise;
    expect(submitResponse.ok()).toBeTruthy();
    expect(submitResponse.request().postDataJSON()).toMatchObject({
      job_type: 'report_data_quality',
      params: { overlap_sample: '5', merged_sample: '5' },
    });
    const job = await submitResponse.json() as { job_id: string };
    createdQuantJobIds.add(job.job_id);

    await waitForJobToFinish(page, job.job_id);
    await page.reload({ waitUntil: 'domcontentloaded' });
    await expect(page.getByText(job.job_id)).toBeVisible({ timeout: 15_000 });
    const previewButton = page.getByRole('button', { name: /预览 data_quality_report_latest\.md/ });
    await expect(previewButton).toBeVisible({ timeout: 15_000 });
    await previewButton.click();
    await expect(page.locator('pre')).toContainText('A-Share Data Quality Report', { timeout: 15_000 });
    await expect(page.locator('pre')).toContainText('Qlib Provider', { timeout: 15_000 });

    expect(consoleWatch.errors).toEqual([]);
  } finally {
    consoleWatch.stop();
  }
});

test('workflow page can launch the anomaly stock report from the browser', async ({ page }) => {
  const consoleWatch = watchConsole(page);
  try {
    await openWorkbench(page, '/workflow');
    await expect(page.getByRole('heading', { name: '量化任务中心' })).toBeVisible();
    await expect(page.getByRole('heading', { name: '生成异常股票和清洗策略' })).toBeVisible({ timeout: 15_000 });

    const responsePromise = page.waitForResponse(
      (response) => response.url().includes('/api/v1/quant/jobs') && response.request().method() === 'POST',
      { timeout: 15_000 },
    );
    await page.getByRole('button', { name: '运行 生成异常股票和清洗策略' }).click();
    const submitResponse = await responsePromise;
    expect(submitResponse.ok()).toBeTruthy();
    expect(submitResponse.request().postDataJSON()).toMatchObject({
      job_type: 'report_data_anomalies',
      params: {},
    });
    const job = await submitResponse.json() as { job_id: string };
    createdQuantJobIds.add(job.job_id);

    await waitForJobToFinish(page, job.job_id);
    await page.reload({ waitUntil: 'domcontentloaded' });
    await expect(page.getByText(job.job_id)).toBeVisible({ timeout: 15_000 });
    const previewButton = page.getByRole('button', { name: /预览 data_cleaning_strategy_latest\.md/ });
    await expect(previewButton).toBeVisible({ timeout: 15_000 });
    await previewButton.click();
    await expect(page.locator('pre')).toContainText('Abnormal Stock List and Cleaning Strategy', { timeout: 15_000 });
    await expect(page.locator('pre')).toContainText('Cleaning Strategy', { timeout: 15_000 });

    expect(consoleWatch.errors).toEqual([]);
  } finally {
    consoleWatch.stop();
  }
});

test('workflow page can launch data governance and capability inventory reports from the browser', async ({ page }) => {
  const consoleWatch = watchConsole(page);
  try {
    await openWorkbench(page, '/workflow');
    await expect(page.getByRole('heading', { name: '量化任务中心' })).toBeVisible();

    const hygieneResponsePromise = page.waitForResponse(
      (response) => response.url().includes('/api/v1/quant/jobs') && response.request().method() === 'POST',
      { timeout: 15_000 },
    );
    await page.getByRole('button', { name: '运行 生成数据治理盘点和清理预案' }).click();
    const hygieneResponse = await hygieneResponsePromise;
    expect(hygieneResponse.ok()).toBeTruthy();
    expect(hygieneResponse.request().postDataJSON()).toMatchObject({
      job_type: 'report_data_hygiene',
      params: {},
    });
    const hygieneJob = await hygieneResponse.json() as { job_id: string };
    createdQuantJobIds.add(hygieneJob.job_id);

    await waitForJobToFinish(page, hygieneJob.job_id);
    await page.reload({ waitUntil: 'domcontentloaded' });
    await expect(page.getByText(hygieneJob.job_id)).toBeVisible({ timeout: 15_000 });
    const hygienePreview = page.getByRole('button', { name: /预览 data_hygiene_inventory_latest\.md/ });
    await expect(hygienePreview).toBeVisible({ timeout: 15_000 });
    await hygienePreview.click();
    await expect(page.locator('pre')).toContainText('A 股量化数据治理 Inventory', { timeout: 15_000 });

    await page.reload({ waitUntil: 'domcontentloaded' });
    const capabilityResponsePromise = page.waitForResponse(
      (response) => response.url().includes('/api/v1/quant/jobs') && response.request().method() === 'POST',
      { timeout: 15_000 },
    );
    await page.getByRole('button', { name: '运行 生成工作台能力盘点' }).click();
    const capabilityResponse = await capabilityResponsePromise;
    expect(capabilityResponse.ok()).toBeTruthy();
    expect(capabilityResponse.request().postDataJSON()).toMatchObject({
      job_type: 'report_workbench_capabilities',
      params: {},
    });
    const capabilityJob = await capabilityResponse.json() as { job_id: string };
    createdQuantJobIds.add(capabilityJob.job_id);

    await waitForJobToFinish(page, capabilityJob.job_id);
    await page.reload({ waitUntil: 'domcontentloaded' });
    await expect(page.getByText(capabilityJob.job_id)).toBeVisible({ timeout: 15_000 });
    const capabilityPreview = page.getByRole('button', { name: /预览 workbench_capability_inventory_latest\.md/ });
    await expect(capabilityPreview).toBeVisible({ timeout: 15_000 });
    await capabilityPreview.click();
    await expect(page.locator('pre')).toContainText('DSA 本地量化工作台能力盘点', { timeout: 15_000 });

    expect(consoleWatch.errors).toEqual([]);
  } finally {
    consoleWatch.stop();
  }
});

test('workflow page can launch a sampled market regime report and preview the cash signal', async ({ page }) => {
  const consoleWatch = watchConsole(page);
  try {
    await openWorkbench(page, '/workflow');
    await expect(page.getByRole('heading', { name: '量化任务中心' })).toBeVisible();
    await expect(page.getByRole('heading', { name: '生成市场状态和现金仓位信号' })).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText(/只读研究报告，不连接券商/)).toBeVisible();

    await page.getByLabel('生成市场状态和现金仓位信号 回看交易日').fill('80');
    await page.getByLabel('生成市场状态和现金仓位信号 最少股票数').fill('2');
    await page.getByLabel('生成市场状态和现金仓位信号 抽样上限').fill('20');

    const responsePromise = page.waitForResponse(
      (response) => response.url().includes('/api/v1/quant/jobs') && response.request().method() === 'POST',
      { timeout: 15_000 },
    );
    await page.getByRole('button', { name: '运行 生成市场状态和现金仓位信号' }).click();
    const submitResponse = await responsePromise;
    expect(submitResponse.ok()).toBeTruthy();
    expect(submitResponse.request().postDataJSON()).toMatchObject({
      job_type: 'report_market_regime',
      params: { lookback_days: '80', min_symbols: '2', limit: '20' },
    });
    const job = await submitResponse.json() as { job_id: string };
    createdQuantJobIds.add(job.job_id);

    await waitForJobToFinish(page, job.job_id);
    await page.reload({ waitUntil: 'domcontentloaded' });
    await expect(page.getByText(job.job_id)).toBeVisible({ timeout: 15_000 });
    const previewButton = page.getByRole('button', { name: /预览 market_regime_latest\.md/ });
    await expect(previewButton).toBeVisible({ timeout: 15_000 });
    await previewButton.click();
    await expect(page.locator('pre')).toContainText('Market Regime Report', { timeout: 15_000 });
    await expect(page.locator('pre')).toContainText('Suggested cash_ratio', { timeout: 15_000 });
    await expect(page.locator('pre')).toContainText('read-only research signal', { timeout: 15_000 });

    expect(consoleWatch.errors).toEqual([]);
  } finally {
    consoleWatch.stop();
  }
});

test('workflow page can launch a sampled cash-control backtest and preview the report', async ({ page }) => {
  const consoleWatch = watchConsole(page);
  try {
    await openWorkbench(page, '/workflow');
    await expect(page.getByRole('heading', { name: '量化任务中心' })).toBeVisible();
    await expect(page.getByRole('heading', { name: '生成现金仓位叠加回测' })).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText(/baseline vs cash-control/)).toBeVisible();

    await page.getByLabel('生成现金仓位叠加回测 回看交易日').fill('80');
    await page.getByLabel('生成现金仓位叠加回测 最少股票数').fill('2');
    await page.getByLabel('生成现金仓位叠加回测 抽样上限').fill('20');

    const responsePromise = page.waitForResponse(
      (response) => response.url().includes('/api/v1/quant/jobs') && response.request().method() === 'POST',
      { timeout: 15_000 },
    );
    await page.getByRole('button', { name: '运行 生成现金仓位叠加回测' }).click();
    const submitResponse = await responsePromise;
    expect(submitResponse.ok()).toBeTruthy();
    expect(submitResponse.request().postDataJSON()).toMatchObject({
      job_type: 'report_cash_control_backtest',
      params: { lookback_days: '80', min_symbols: '2', limit: '20' },
    });
    const job = await submitResponse.json() as { job_id: string };
    createdQuantJobIds.add(job.job_id);

    await waitForJobToFinish(page, job.job_id);
    await page.reload({ waitUntil: 'domcontentloaded' });
    await expect(page.getByText(job.job_id)).toBeVisible({ timeout: 15_000 });
    const previewButton = page.getByRole('button', { name: /预览 cash_control_backtest_latest\.md/ });
    await expect(previewButton).toBeVisible({ timeout: 15_000 });
    await previewButton.click();
    await expect(page.locator('pre')).toContainText('Cash-Control Backtest', { timeout: 15_000 });
    await expect(page.locator('pre')).toContainText('read-only research report', { timeout: 15_000 });
    await expect(page.locator('pre')).toContainText('不连接券商', { timeout: 15_000 });

    expect(consoleWatch.errors).toEqual([]);
  } finally {
    consoleWatch.stop();
  }
});

test('workflow page can launch a strategy optimization dry-run and preview the report', async ({ page }) => {
  const consoleWatch = watchConsole(page);
  try {
    await openWorkbench(page, '/workflow');
    await expect(page.getByRole('heading', { name: '量化任务中心' })).toBeVisible();
    await expect(page.getByRole('heading', { name: '策略优化预演' })).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText(/不训练、不连接券商、不自动下单/)).toBeVisible();

    await page.getByLabel('策略优化预演 阶段').selectOption('4');
    await page.getByLabel('策略优化预演 TopK').fill('30');

    const responsePromise = page.waitForResponse(
      (response) => response.url().includes('/api/v1/quant/jobs') && response.request().method() === 'POST',
      { timeout: 15_000 },
    );
    await page.getByRole('button', { name: '运行 策略优化预演' }).click();
    const submitResponse = await responsePromise;
    expect(submitResponse.ok()).toBeTruthy();
    expect(submitResponse.request().postDataJSON()).toMatchObject({
      job_type: 'optimize_strategy',
      params: { dry_run: true, phase: '4', topk: '30' },
    });
    const job = await submitResponse.json() as { job_id: string };
    createdQuantJobIds.add(job.job_id);

    await waitForJobToFinish(page, job.job_id);
    await page.reload({ waitUntil: 'domcontentloaded' });
    await expect(page.getByText(job.job_id)).toBeVisible({ timeout: 15_000 });
    const previewButton = page.getByRole('button', { name: /预览 optimization_results_latest\.md/ });
    await expect(previewButton).toBeVisible({ timeout: 15_000 });
    await previewButton.click();
    await expect(page.locator('pre')).toContainText('Qlib 策略优化实验报告', { timeout: 15_000 });
    await expect(page.locator('pre')).toContainText('只读策略优化预演', { timeout: 15_000 });
    await expect(page.locator('pre')).toContainText('不自动下单', { timeout: 15_000 });

    expect(consoleWatch.errors).toEqual([]);
  } finally {
    consoleWatch.stop();
  }
});

test('workflow page can launch the Qlib prediction summary and preview the report', async ({ page }) => {
  const consoleWatch = watchConsole(page);
  try {
    await openWorkbench(page, '/workflow');
    await expect(page.getByRole('heading', { name: '量化任务中心' })).toBeVisible();
    await expect(page.getByRole('heading', { name: '生成 Qlib 预测摘要' })).toBeVisible({ timeout: 15_000 });

    await page.getByLabel('生成 Qlib 预测摘要 TopK').fill('3');

    const responsePromise = page.waitForResponse(
      (response) => response.url().includes('/api/v1/quant/jobs') && response.request().method() === 'POST',
      { timeout: 15_000 },
    );
    await page.getByRole('button', { name: '运行 生成 Qlib 预测摘要' }).click();
    const submitResponse = await responsePromise;
    expect(submitResponse.ok()).toBeTruthy();
    expect(submitResponse.request().postDataJSON()).toMatchObject({
      job_type: 'report_qlib_predictions',
      params: { topk: '3' },
    });
    const job = await submitResponse.json() as { job_id: string };
    createdQuantJobIds.add(job.job_id);

    await waitForJobToFinish(page, job.job_id);
    await page.reload({ waitUntil: 'domcontentloaded' });
    await expect(page.getByText(job.job_id)).toBeVisible({ timeout: 15_000 });
    const previewButton = page.getByRole('button', { name: /预览 qlib_predictions_latest\.md/ });
    await expect(previewButton).toBeVisible({ timeout: 15_000 });
    await previewButton.click();
    await expect(page.locator('pre')).toContainText('Qlib Prediction Summary', { timeout: 15_000 });
    await expect(page.locator('pre')).toContainText('不自动下单', { timeout: 15_000 });

    expect(consoleWatch.errors).toEqual([]);
  } finally {
    consoleWatch.stop();
  }
});

test('workflow page exposes the china-astock report-only comparison task without launching it', async ({ page }) => {
  const consoleWatch = watchConsole(page);
  try {
    await openWorkbench(page, '/workflow');
    await expect(page.getByRole('heading', { name: '量化任务中心' })).toBeVisible();
    await expect(page.getByRole('heading', { name: '生成 china-astock 策略对比' })).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText('读取 china-astock-quant 既有搜索和赢家评估结果，生成策略对比 CSV/Markdown 报告；只读报告，不连接券商或交易接口。')).toBeVisible();
    await expect(page.getByRole('button', { name: '运行 生成 china-astock 策略对比' })).toBeVisible();

    const statusResponse = await page.request.get('/api/v1/quant/status');
    expect(statusResponse.ok()).toBeTruthy();
    const status = await statusResponse.json() as {
      available_job_types?: Record<string, { allowed_params?: string[] }>;
    };
    expect(status.available_job_types?.run_china_astock_comparison?.allowed_params).toEqual([]);
    expect(consoleWatch.errors).toEqual([]);
  } finally {
    consoleWatch.stop();
  }
});

test('workflow page can launch the china-astock technical indicator report from the browser', async ({ page }) => {
  const consoleWatch = watchConsole(page);
  try {
    await openWorkbench(page, '/workflow');
    await expect(page.getByRole('heading', { name: '量化任务中心' })).toBeVisible();
    await expect(page.getByRole('heading', { name: '生成 china-astock 技术指标报告' })).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText(/复用 china-astock-quant 的 SMA\/EMA\/RSI\/ATR\/Bollinger/)).toBeVisible();

    await page.getByLabel('生成 china-astock 技术指标报告 股票代码').fill('000001,600519');
    await page.getByLabel('生成 china-astock 技术指标报告 回看交易日').fill('60');
    await page.getByLabel('生成 china-astock 技术指标报告 股票数量上限').fill('2');

    const responsePromise = page.waitForResponse(
      (response) => response.url().includes('/api/v1/quant/jobs') && response.request().method() === 'POST',
      { timeout: 15_000 },
    );
    await page.getByRole('button', { name: '运行 生成 china-astock 技术指标报告' }).click();
    const submitResponse = await responsePromise;
    expect(submitResponse.ok()).toBeTruthy();
    expect(submitResponse.request().postDataJSON()).toMatchObject({
      job_type: 'report_china_astock_indicators',
      params: { symbols: '000001,600519', lookback_days: '60', limit: '2' },
    });
    const job = await submitResponse.json() as { job_id: string };
    createdQuantJobIds.add(job.job_id);

    await waitForJobToFinish(page, job.job_id);
    await page.reload({ waitUntil: 'domcontentloaded' });
    await expect(page.getByText(job.job_id)).toBeVisible({ timeout: 15_000 });
    const previewButton = page.getByRole('button', { name: /预览 china_astock_indicator_report_latest\.md/ });
    await expect(previewButton).toBeVisible({ timeout: 15_000 });
    await previewButton.click();
    await expect(page.locator('pre')).toContainText('china-astock 技术指标报告', { timeout: 15_000 });
    await expect(page.locator('pre')).toContainText('只读研究报告', { timeout: 15_000 });
    await expect(page.locator('pre')).toContainText('不自动下单', { timeout: 15_000 });
    await expect(page.locator('pre')).toContainText('000001', { timeout: 15_000 });

    expect(consoleWatch.errors).toEqual([]);
  } finally {
    consoleWatch.stop();
  }
});

test('workflow page can select the cleaned provider for training without launching it', async ({ page }) => {
  const consoleWatch = watchConsole(page);
  try {
    await openWorkbench(page, '/workflow');
    await expect(page.getByRole('heading', { name: '量化任务中心' })).toBeVisible();

    const providerSelect = page.getByLabel('训练 Qlib LightGBM 基线 数据版本');
    await expect(providerSelect).toBeVisible({ timeout: 15_000 });
    await providerSelect.selectOption('cleaned');
    await expect(providerSelect).toHaveValue('cleaned');

    const statusResponse = await page.request.get('/api/v1/quant/status');
    expect(statusResponse.ok()).toBeTruthy();
    const status = await statusResponse.json() as {
      available_job_types?: Record<string, { allowed_params?: string[] }>;
    };
    expect(status.available_job_types?.train_baseline_lgb?.allowed_params).toContain('provider');
    expect(consoleWatch.errors).toEqual([]);
  } finally {
    consoleWatch.stop();
  }
});

test('workflow page rejects invalid job parameters before creating a quant job', async ({ page }) => {
  const consoleWatch = watchConsole(page);
  try {
    await openWorkbench(page, '/workflow');
    await expect(page.getByRole('heading', { name: '量化任务中心' })).toBeVisible();

    await page.getByLabel('准备 Baostock 增量清单 股票数量上限').fill('0');
    const responsePromise = page.waitForResponse(
      (response) => response.url().includes('/api/v1/quant/jobs') && response.request().method() === 'POST',
      { timeout: 15_000 },
    );
    await page.getByRole('button', { name: '运行 准备 Baostock 增量清单' }).click();
    const submitResponse = await responsePromise;
    expect(submitResponse.status()).toBe(400);
    const payload = await submitResponse.json() as { error?: string; message?: string };
    expect(payload.error).toBe('invalid_params');
    expect(payload.message).toContain('Param limit must be >= 1.');

    await expect(page.getByText('工作流接口异常')).toBeVisible({ timeout: 10_000 });
    await expect(page.getByText('Param limit must be >= 1.')).toBeVisible({ timeout: 10_000 });
    expect(consoleWatch.errors.filter((message) => !message.includes('status of 400'))).toEqual([]);
  } finally {
    consoleWatch.stop();
  }
});

test('workflow page opens completed quant job artifact links in a browser tab', async ({ page }) => {
  const consoleWatch = watchConsole(page);
  const { jobId } = createTemporaryQuantArtifactJob();
  const artifactName = `${e2ePrefix}_artifact_report.md`;

  try {
    await openWorkbench(page, '/workflow');
    await expect(page.getByText(jobId)).toBeVisible({ timeout: 15_000 });

    const artifactLink = page.getByRole('link', { name: artifactName });
    await expect(artifactLink).toBeVisible({ timeout: 15_000 });
    await expect(artifactLink).toHaveAttribute(
      'href',
      new RegExp(`/api/v1/quant/jobs/${jobId}/artifacts/0$`),
    );

    const [artifactPage] = await Promise.all([
      page.waitForEvent('popup'),
      artifactLink.click(),
    ]);
    await artifactPage.waitForLoadState('domcontentloaded');
    await expect(artifactPage.locator('body')).toContainText('DSA E2E Artifact', { timeout: 15_000 });
    await expect(artifactPage.locator('body')).toContainText(`${e2ePrefix}_artifact_marker`);
    expect(new URL(artifactPage.url()).pathname).toBe(`/api/v1/quant/jobs/${jobId}/artifacts/0`);
    await artifactPage.close();

    expect(consoleWatch.errors).toEqual([]);
  } finally {
    consoleWatch.stop();
  }
});

test('workflow page previews a completed quant text artifact inline', async ({ page }) => {
  const consoleWatch = watchConsole(page);
  const { jobId } = createTemporaryQuantArtifactJob();
  const artifactName = `${e2ePrefix}_artifact_report.md`;

  try {
    await openWorkbench(page, '/workflow');
    await expect(page.getByText(jobId)).toBeVisible({ timeout: 15_000 });

    await page.getByRole('button', { name: `预览 ${artifactName}` }).click();

    await expect(page.getByRole('heading', { name: '产物预览' })).toBeVisible({ timeout: 15_000 });
    await expect(page.locator('pre')).toContainText('DSA E2E Artifact', { timeout: 15_000 });
    await expect(page.locator('pre')).toContainText(`${e2ePrefix}_artifact_marker`);
    expect(consoleWatch.errors).toEqual([]);
  } finally {
    consoleWatch.stop();
  }
});

test('workflow page can cancel a running quant job from the browser', async ({ page }) => {
  const consoleWatch = watchConsole(page);
  const runningJob = createTemporaryRunningQuantJob();

  try {
    await openWorkbench(page, '/workflow');
    await expect(page.getByText(runningJob.jobId)).toBeVisible({ timeout: 15_000 });
    await expect(page.getByRole('button', { name: '取消 检查 Qlib 数据' })).toBeVisible({ timeout: 15_000 });

    const cancelResponsePromise = page.waitForResponse(
      (response) => response.url().includes(`/api/v1/quant/jobs/${runningJob.jobId}/cancel`)
        && response.request().method() === 'POST',
      { timeout: 15_000 },
    );
    await page.getByRole('button', { name: '取消 检查 Qlib 数据' }).click();
    const cancelResponse = await cancelResponsePromise;
    expect(cancelResponse.ok()).toBeTruthy();
    const cancelled = await cancelResponse.json() as { status?: string; error?: string };
    expect(cancelled.status).toBe('cancelled');
    expect(cancelled.error).toContain('Cancelled by user');

    await expect.poll(() => isPidAlive(runningJob.pid), { timeout: 15_000 }).toBe(false);
    await expect(page.getByTestId('workflow-job-check_qlib_data').getByText('已取消')).toBeVisible({ timeout: 15_000 });
    const detail = await page.request.get(`/api/v1/quant/jobs/${runningJob.jobId}`);
    expect(detail.ok()).toBeTruthy();
    expect((await detail.json() as { status?: string }).status).toBe('cancelled');
    expect(consoleWatch.errors).toEqual([]);
  } finally {
    consoleWatch.stop();
  }
});

test('research page can archive and permanently delete temporary experiment records', async ({ page }) => {
  const consoleWatch = watchConsole(page);
  const archiveRunId = `${e2ePrefix}_archive_run`;
  const permanentRunId = `${e2ePrefix}_permanent_run`;
  createTemporaryExperimentRun(archiveRunId, false);
  createTemporaryExperimentRun(permanentRunId, true);

  try {
    await openWorkbench(page, '/research');
    await expect(page.getByText(`Run ${archiveRunId}`)).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText(`Run ${permanentRunId}`)).toBeVisible({ timeout: 15_000 });

    await page.getByRole('button', { name: `归档实验 ${archiveRunId}` }).click();
    await expect(page.getByText(`Run ${archiveRunId}`)).not.toBeVisible({ timeout: 15_000 });

    page.once('dialog', async (dialog) => {
      expect(dialog.message()).toContain('永久删除');
      await dialog.accept();
    });
    await page.getByRole('button', { name: `永久删除实验 ${permanentRunId}` }).click();
    await expect(page.getByText(`Run ${permanentRunId}`)).not.toBeVisible({ timeout: 15_000 });

    const reportsRoot = getQuantReportsRoot();
    expect(fs.readFileSync(path.join(reportsRoot, 'experiment_runs.jsonl'), 'utf-8')).not.toContain(permanentRunId);
    expect(fs.readFileSync(path.join(reportsRoot, 'experiment_log.md'), 'utf-8')).not.toContain(permanentRunId);
    expect(consoleWatch.errors).toEqual([]);
  } finally {
    consoleWatch.stop();
  }
});

test('backtest page can run a narrow DSA validation and switch to the Qlib backtest tab', async ({ page }) => {
  const consoleWatch = watchConsole(page);
  try {
    await openWorkbench(page, '/backtest');
    await expect(page.getByRole('button', { name: 'DSA分析准确性回测' })).toBeVisible();

    await page.getByPlaceholder('按股票代码筛选（留空表示全部）').fill('000000');
    await page.getByPlaceholder('10').fill('1');
    await page.getByRole('button', { name: '1 日验证' }).click();
    const runResponsePromise = page.waitForResponse(
      (response) => response.url().includes('/api/v1/backtest/run') && response.request().method() === 'POST',
      { timeout: 30_000 },
    );
    await page.getByRole('button', { name: '运行回测' }).click();
    const runResponse = await runResponsePromise;
    expect(runResponse.ok()).toBeTruthy();
    await expect(page.getByText('已处理:')).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText('已保存:')).toBeVisible({ timeout: 15_000 });

    await page.getByRole('button', { name: 'Qlib模型/策略回测' }).click();
    await expect(page.getByRole('heading', { name: 'Qlib 模型/策略回测' })).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText('模型结果 ≠ 交易建议')).toBeVisible({ timeout: 15_000 });
    await expect(page.getByRole('heading', { name: '实验复现与产物' })).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText('pred.pkl')).toBeVisible({ timeout: 15_000 });
    await expect(page.getByRole('heading', { name: '现金仓位叠加回测' })).toBeVisible({ timeout: 15_000 });
    expect(consoleWatch.errors).toEqual([]);
  } finally {
    consoleWatch.stop();
  }
});

test('home page can submit an E2E fake analysis and show the completed report from the browser', async ({ page }) => {
  test.skip(
    process.env.DSA_WEB_E2E_FAKE_ANALYSIS !== '1' && process.env.DSA_E2E_FAKE_ANALYSIS !== '1',
    'Set DSA_WEB_E2E_FAKE_ANALYSIS=1 and run against a backend started with DSA_E2E_FAKE_ANALYSIS=1.',
  );

  const consoleWatch = watchConsole(page);
  const stockCode = '689997';
  const stockName = `E2E${stockCode}`;
  let queryId = '';

  try {
    cleanupE2eFakeAnalysisHistoryDatabase();
    await openWorkbench(page, '/');
    await expect(page.getByTestId('home-dashboard')).toBeVisible({ timeout: 15_000 });

    await page.getByPlaceholder(/输入股票代码或名称/).fill(stockCode);
    const analyzeResponsePromise = page.waitForResponse(
      (response) => response.url().includes('/api/v1/analysis/analyze') && response.request().method() === 'POST',
      { timeout: 15_000 },
    );
    await page.getByRole('button', { name: '分析', exact: true }).click();
    const analyzeResponse = await analyzeResponsePromise;
    expect(analyzeResponse.status()).toBe(202);
    const accepted = await analyzeResponse.json() as { task_id?: string; taskId?: string };
    const taskId = accepted.task_id || accepted.taskId || '';
    expect(taskId).toBeTruthy();

    const taskStatus = await waitForAnalysisTaskToComplete(page, taskId);
    queryId = taskStatus.result?.query_id || taskStatus.result?.queryId || '';
    expect(queryId).toMatch(/^dsa_e2e_fake_analysis_/);
    createdAnalysisHistoryQueryIds.add(queryId);
    const summary = taskStatus.result?.report?.summary?.analysis_summary
      || taskStatus.result?.report?.summary?.analysisSummary
      || '';
    expect(summary).toContain(`dsa_e2e_fake_analysis 本地浏览器验证报告：${stockCode}`);

    await expect(
      page.getByRole('button', { name: `${stockName} ${stockCode} 历史记录` }),
    ).toBeVisible({ timeout: 30_000 });
    await page.getByRole('button', { name: `${stockName} ${stockCode} 历史记录` }).click();
    await expect(page.getByText(`dsa_e2e_fake_analysis 本地浏览器验证报告：${stockCode}`)).toBeVisible({ timeout: 15_000 });

    const databasePath = getDsaDatabasePath();
    const persistedCount = execFileSync(
      'sqlite3',
      [databasePath, `SELECT COUNT(*) FROM analysis_history WHERE query_id = ${sqlString(queryId)};`],
      { encoding: 'utf-8', stdio: 'pipe' },
    ).trim();
    expect(persistedCount).toBe('1');
    expect(consoleWatch.errors).toEqual([]);
  } finally {
    if (queryId) {
      cleanupAnalysisHistoryDatabase([queryId]);
    }
    cleanupE2eFakeAnalysisHistoryDatabase();
    consoleWatch.stop();
  }
});

test('home page can open a persisted history report, show markdown, and delete the stock history', async ({ page }) => {
  const consoleWatch = watchConsole(page);
  const history = createTemporaryAnalysisHistory();

  try {
    await openWorkbench(page, '/');
    await expect(page.getByRole('button', { name: `${history.stockName} ${history.stockCode} 历史记录` })).toBeVisible({ timeout: 15_000 });

    await page.getByRole('button', { name: `${history.stockName} ${history.stockCode} 历史记录` }).click();
    await expect(page.getByText(history.summary)).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText(history.stockCode).first()).toBeVisible();

    await page.getByRole('button', { name: '完整分析报告' }).click();
    const reportDialog = page.locator('[role="dialog"]', {
      has: page.getByRole('heading', { name: history.stockName, exact: true }),
    });
    await expect(reportDialog.getByRole('heading', { name: history.stockName, exact: true })).toBeVisible({ timeout: 15_000 });
    await expect(reportDialog.getByText(history.summary)).toBeVisible({ timeout: 15_000 });
    const markdownResponse = await page.request.get(`/api/v1/history/${history.recordId}/markdown`);
    expect(markdownResponse.ok()).toBeTruthy();
    expect((await markdownResponse.json() as { content?: string }).content).toContain(history.summary);
    await reportDialog.getByRole('button', { name: '关闭', exact: true }).click();
    await expect(reportDialog).not.toBeVisible({ timeout: 15_000 });

    const deleteResponsePromise = page.waitForResponse(
      (response) => response.url().includes(`/api/v1/history/by-code/${history.stockCode}`) && response.request().method() === 'DELETE',
      { timeout: 15_000 },
    );
    await page.getByRole('button', { name: `删除 ${history.stockName} 历史记录` }).click();
    const deleteResponse = await deleteResponsePromise;
    expect(deleteResponse.ok()).toBeTruthy();
    await expect(page.getByRole('button', { name: `${history.stockName} ${history.stockCode} 历史记录` })).not.toBeVisible({ timeout: 15_000 });

    const databasePath = getDsaDatabasePath();
    const remaining = execFileSync(
      'sqlite3',
      [databasePath, `SELECT COUNT(*) FROM analysis_history WHERE query_id = ${sqlString(history.queryId)};`],
      { encoding: 'utf-8', stdio: 'pipe' },
    ).trim();
    expect(remaining).toBe('0');
    expect(consoleWatch.errors).toEqual([]);
  } finally {
    consoleWatch.stop();
  }
});

test('chat page can export and permanently delete a persisted conversation', async ({ page }) => {
  const consoleWatch = watchConsole(page);
  const chat = createTemporaryChatSession();

  try {
    await openWorkbench(page, '/chat');
    await expect(page.getByTestId('chat-workspace')).toBeVisible({ timeout: 15_000 });

    const titleProbe = chat.userContent.slice(0, 45);
    const sessionRow = page.locator('.session-item-row', { hasText: titleProbe });
    await expect(sessionRow).toBeVisible({ timeout: 15_000 });
    await sessionRow.getByRole('button', { name: new RegExp(`切换到对话 ${escapeRegExp(titleProbe)}`) }).click();
    await expect(page.locator('p').filter({ hasText: chat.userContent }).last()).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText(`${e2ePrefix} 问股回复`)).toBeVisible({ timeout: 15_000 });

    const downloadPromise = page.waitForEvent('download');
    await page.getByRole('button', { name: '导出会话为 Markdown 文件' }).click();
    const download = await downloadPromise;
    expect(download.suggestedFilename()).toMatch(/^问股会话_\d{8}_\d{4}\.md$/);
    const downloadPath = await download.path();
    expect(downloadPath).toBeTruthy();
    expect(fs.readFileSync(downloadPath!, 'utf-8')).toContain(chat.userContent);
    expect(fs.readFileSync(downloadPath!, 'utf-8')).toContain(chat.assistantContent);

    await sessionRow.getByRole('button', { name: new RegExp(`删除对话 ${escapeRegExp(titleProbe)}`) }).click();
    const deleteDialog = page.getByRole('dialog', { name: '删除对话' });
    await expect(deleteDialog).toBeVisible({ timeout: 10_000 });
    const deleteResponsePromise = page.waitForResponse(
      (response) => response.url().includes(`/api/v1/agent/chat/sessions/${encodeURIComponent(chat.sessionId)}`)
        && response.request().method() === 'DELETE',
      { timeout: 15_000 },
    );
    await deleteDialog.getByRole('button', { name: '删除', exact: true }).click();
    const deleteResponse = await deleteResponsePromise;
    expect(deleteResponse.ok()).toBeTruthy();
    await expect(sessionRow).not.toBeVisible({ timeout: 15_000 });

    const databasePath = getDsaDatabasePath();
    const remaining = execFileSync(
      'sqlite3',
      [
        databasePath,
        [
          `SELECT COUNT(*) FROM conversation_messages WHERE session_id = ${sqlString(chat.sessionId)};`,
          `SELECT COUNT(*) FROM conversation_summaries WHERE session_id = ${sqlString(chat.sessionId)};`,
          `SELECT COUNT(*) FROM agent_provider_turns WHERE session_id = ${sqlString(chat.sessionId)};`,
        ].join(' '),
      ],
      { encoding: 'utf-8', stdio: 'pipe' },
    ).trim().split(/\r?\n/);
    expect(remaining).toEqual(['0', '0', '0']);
    expect(consoleWatch.errors).toEqual([]);
  } finally {
    consoleWatch.stop();
  }
});

test('home and settings pages expose safe self-service controls without launching long tasks', async ({ page }) => {
  const consoleWatch = watchConsole(page);
  try {
    await openWorkbench(page, '/');
    await expect(page.getByTestId('home-dashboard')).toBeVisible({ timeout: 15_000 });

    const stockInput = page.getByPlaceholder(/输入股票代码或名称/);
    await stockInput.fill('600519x');
    await page.getByRole('button', { name: '分析', exact: true }).click();
    await expect(page.getByText('请输入有效的股票代码或股票名称')).toBeVisible({ timeout: 10_000 });

    const notifyCheckbox = page.getByLabel('推送通知');
    const notifyInitiallyChecked = await notifyCheckbox.isChecked();
    await notifyCheckbox.setChecked(!notifyInitiallyChecked);
    await expect(notifyCheckbox).toBeChecked({ checked: !notifyInitiallyChecked });
    await notifyCheckbox.setChecked(notifyInitiallyChecked);
    await expect(notifyCheckbox).toBeChecked({ checked: notifyInitiallyChecked });

    const skillsResponse = await page.request.get('/api/v1/agent/skills');
    if (skillsResponse.ok()) {
      const skillsData = await skillsResponse.json() as { skills?: unknown[] };
      if ((skillsData.skills || []).length > 0) {
        const strategyButton = page.getByRole('button', { name: /策略/ });
        await expect(strategyButton).toBeVisible({ timeout: 10_000 });
        await strategyButton.click();
        await expect(page.getByRole('menu')).toBeVisible();
        await expect(page.getByRole('menuitemradio').first()).toBeVisible();
        await page.keyboard.press('Escape');
        await expect(page.getByRole('menu')).not.toBeVisible();
      }
    }

    await openWorkbench(page, '/settings');
    await expect(page.getByRole('heading', { name: '系统设置' })).toBeVisible({ timeout: 15_000 });
    await page.getByRole('button', { name: /基础设置/ }).click();
    await expect(page.getByRole('heading', { name: '智能导入' })).toBeVisible({ timeout: 15_000 });
    await expect(page.getByRole('button', { name: '选择图片' })).toBeVisible();
    await expect(page.getByRole('button', { name: '选择文件' })).toBeVisible();

    await page.getByRole('button', { name: /查看 自选股列表 配置说明/ }).click();
    const helpDialog = page.getByRole('dialog', { name: '自选股列表' });
    await expect(helpDialog).toBeVisible({ timeout: 10_000 });
    await expect(helpDialog).toContainText('STOCK_LIST');
    await helpDialog.getByRole('button', { name: '关闭配置说明' }).click();
    await expect(helpDialog).not.toBeVisible();

    await page.getByRole('button', { name: /AI 模型/ }).click();
    await expect(page.getByRole('heading', { name: '模型接入' })).toBeVisible({ timeout: 15_000 });

    expect(consoleWatch.errors).toEqual([]);
  } finally {
    consoleWatch.stop();
  }
});

test('settings page can save and restore the stock list configuration from the browser', async ({ page }) => {
  const consoleWatch = watchConsole(page);
  const originalStockList = await getSystemConfigValue(page.request, 'STOCK_LIST');
  const nextStockList = originalStockList.replace(/\s/g, '') === '600519,300750'
    ? '600519,002594'
    : '600519,300750';
  pendingSystemConfigRestores.set('STOCK_LIST', originalStockList);

  try {
    await openWorkbench(page, '/settings');
    await expect(page.getByRole('heading', { name: '系统设置' })).toBeVisible({ timeout: 15_000 });

    const stockListInput = page.getByLabel('自选股列表', { exact: true });
    await expect(stockListInput).toBeVisible({ timeout: 15_000 });
    await stockListInput.fill(nextStockList);
    await expect(page.getByRole('button', { name: '保存配置 (1)' })).toBeEnabled({ timeout: 10_000 });

    const saveResponsePromise = page.waitForResponse(
      (response) => response.url().includes('/api/v1/system/config')
        && response.request().method() === 'PUT',
      { timeout: 15_000 },
    );
    await page.getByRole('button', { name: '保存配置 (1)' }).click();
    const saveResponse = await saveResponsePromise;
    expect(saveResponse.ok()).toBeTruthy();
    await expect(page.getByText('配置已更新')).toBeVisible({ timeout: 15_000 });

    await page.reload({ waitUntil: 'domcontentloaded' });
    await expect(page.getByLabel('自选股列表', { exact: true })).toHaveValue(nextStockList, { timeout: 15_000 });
    expect(await getSystemConfigValue(page.request, 'STOCK_LIST')).toBe(nextStockList);
    expect(consoleWatch.errors).toEqual([]);
  } finally {
    await updateSystemConfigValue(page.request, 'STOCK_LIST', originalStockList).catch(() => undefined);
    pendingSystemConfigRestores.delete('STOCK_LIST');
    consoleWatch.stop();
  }
});

test('settings intelligent import can parse pasted text and merge into watchlist from the browser', async ({ page }) => {
  const consoleWatch = watchConsole(page);
  const originalStockList = await getSystemConfigValue(page.request, 'STOCK_LIST');
  const baseStockList = '000001';
  const importedStockList = `${baseStockList},600519,300750`;
  pendingSystemConfigRestores.set('STOCK_LIST', originalStockList);

  try {
    await updateSystemConfigValue(page.request, 'STOCK_LIST', baseStockList);
    await openWorkbench(page, '/settings');
    await expect(page.getByRole('heading', { name: '系统设置' })).toBeVisible({ timeout: 15_000 });
    await page.getByRole('button', { name: /基础设置/ }).click();
    await expect(page.getByRole('heading', { name: '智能导入' })).toBeVisible({ timeout: 15_000 });

    await page.getByPlaceholder('或粘贴 CSV/Excel 复制的文本...').fill('600519,贵州茅台\n300750,宁德时代');
    const parseResponsePromise = page.waitForResponse(
      (response) => response.url().includes('/api/v1/stocks/parse-import')
        && response.request().method() === 'POST',
      { timeout: 15_000 },
    );
    await page.getByRole('button', { name: '解析', exact: true }).click();
    const parseResponse = await parseResponsePromise;
    expect(parseResponse.ok()).toBeTruthy();

    await expect(page.getByText('600519', { exact: true })).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText('300750', { exact: true })).toBeVisible({ timeout: 15_000 });
    await page.getByRole('button', { name: '全选当前' }).click();
    await expect(page.getByText('共 2 条可合并，已勾选 2 条')).toBeVisible({ timeout: 10_000 });

    const saveResponsePromise = page.waitForResponse(
      (response) => response.url().includes('/api/v1/system/config')
        && response.request().method() === 'PUT',
      { timeout: 15_000 },
    );
    await page.getByRole('button', { name: '合并到自选股' }).click();
    const saveResponse = await saveResponsePromise;
    expect(saveResponse.ok()).toBeTruthy();
    await expect(page.getByText('共 2 条可合并，已勾选 2 条')).not.toBeVisible({ timeout: 15_000 });

    expect(await getSystemConfigValue(page.request, 'STOCK_LIST')).toBe(importedStockList);
    await page.reload({ waitUntil: 'domcontentloaded' });
    await expect(page.getByLabel('自选股列表', { exact: true })).toHaveValue(importedStockList, { timeout: 15_000 });
    expect(consoleWatch.errors).toEqual([]);
  } finally {
    await updateSystemConfigValue(page.request, 'STOCK_LIST', originalStockList).catch(() => undefined);
    pendingSystemConfigRestores.delete('STOCK_LIST');
    consoleWatch.stop();
  }
});

test('settings env backup export/import is either protected or works from the browser', async ({ page }) => {
  const consoleWatch = watchConsole(page);
  const originalStockList = await getSystemConfigValue(page.request, 'STOCK_LIST');
  const importedStockList = originalStockList.replace(/\s/g, '') === '600519,300750'
    ? '600519,002594'
    : '600519,300750';
  const exportProbe = await page.request.get('/api/v1/system/config/export');
  const canUseEnvBackup = exportProbe.ok();

  if (canUseEnvBackup) {
    pendingSystemConfigRestores.set('STOCK_LIST', originalStockList);
    await page.addInitScript(() => {
      window.dsaDesktop = { version: 'dsa-e2e' };
    });
  }

  try {
    await openWorkbench(page, '/settings');
    await expect(page.getByRole('heading', { name: '系统设置' })).toBeVisible({ timeout: 15_000 });
    await page.getByRole('button', { name: /系统设置 管理调度/ }).click();
    await expect(page.getByRole('heading', { name: '配置备份' })).toBeVisible({ timeout: 15_000 });

    const exportButton = page.getByRole('button', { name: '导出 .env' });
    const importButton = page.getByRole('button', { name: '导入 .env' });

    if (!canUseEnvBackup) {
      await expect(page.getByText(/当前 Web 端未开启管理员鉴权/)).toBeVisible({ timeout: 15_000 });
      await expect(exportButton).toBeDisabled();
      await expect(importButton).toBeDisabled();
      expect(exportProbe.status()).toBe(403);
      expect(consoleWatch.errors).toEqual([]);
      return;
    }

    await expect(exportButton).toBeEnabled({ timeout: 15_000 });
    const downloadPromise = page.waitForEvent('download');
    await exportButton.click();
    const download = await downloadPromise;
    expect(download.suggestedFilename()).toMatch(/^dsa-desktop-env_.*\.env$/);
    const downloadPath = await download.path();
    expect(downloadPath).toBeTruthy();
    const downloadedContent = fs.readFileSync(downloadPath as string, 'utf-8');
    expect(downloadedContent).toContain('STOCK_LIST=');
    await expect(page.getByText('已导出当前已保存的 .env 备份。')).toBeVisible({ timeout: 15_000 });

    const importResponsePromise = page.waitForResponse(
      (response) => response.url().includes('/api/v1/system/config/import')
        && response.request().method() === 'POST',
      { timeout: 15_000 },
    );
    await page.locator('input[type="file"][accept=".env,.txt"]').setInputFiles({
      name: 'dsa-e2e-settings-backup.env',
      mimeType: 'text/plain',
      buffer: Buffer.from(`STOCK_LIST=${importedStockList}\n`, 'utf-8'),
    });
    const importResponse = await importResponsePromise;
    expect(importResponse.ok()).toBeTruthy();
    await expect(page.getByText('已导入 .env 备份并重新加载配置。')).toBeVisible({ timeout: 15_000 });
    expect(await getSystemConfigValue(page.request, 'STOCK_LIST')).toBe(importedStockList);

    await page.getByRole('button', { name: /基础设置/ }).click();
    await expect(page.getByLabel('自选股列表', { exact: true })).toHaveValue(importedStockList, { timeout: 15_000 });
    expect(consoleWatch.errors).toEqual([]);
  } finally {
    if (canUseEnvBackup) {
      await updateSystemConfigValue(page.request, 'STOCK_LIST', originalStockList).catch(() => undefined);
      pendingSystemConfigRestores.delete('STOCK_LIST');
    }
    consoleWatch.stop();
  }
});

test('screening page can run an AlphaSift screen from the browser when the adapter is available', async ({ page }) => {
  test.setTimeout(210_000);
  const consoleWatch = watchConsole(page);
  const originalAlphaSiftEnabled = await getSystemConfigValue(page.request, 'ALPHASIFT_ENABLED');
  pendingSystemConfigRestores.set('ALPHASIFT_ENABLED', originalAlphaSiftEnabled);

  try {
    await updateSystemConfigValue(page.request, 'ALPHASIFT_ENABLED', 'false');
    await expect.poll(async () => (await getAlphaSiftStatus(page.request)).enabled).toBe(false);

    await openWorkbench(page, '/screening');
    await expect(page.getByRole('heading', { name: 'AlphaSift 选股' })).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText('AlphaSift 未开启')).toBeVisible({ timeout: 15_000 });
    await expect(page.getByRole('button', { name: '开启 AlphaSift' })).toBeVisible();
    await expect(page.getByRole('button', { name: '运行选股' })).toBeDisabled();

    await updateSystemConfigValue(page.request, 'ALPHASIFT_ENABLED', 'true');
    await expect.poll(async () => {
      const status = await getAlphaSiftStatus(page.request);
      return `${status.enabled ? 'enabled' : 'disabled'}:${status.available ? 'available' : 'unavailable'}`;
    }).toMatch(/^enabled:(available|unavailable)$/);

    await page.reload({ waitUntil: 'domcontentloaded' });
    await expect(page.getByRole('heading', { name: 'AlphaSift 选股' })).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText('AlphaSift 未开启')).not.toBeVisible({ timeout: 15_000 });

    const currentStatus = await getAlphaSiftStatus(page.request);
    const runButton = page.getByRole('button', { name: '运行选股' });
    if (currentStatus.available) {
      await expect(page.getByText('选股已开启')).toBeVisible({ timeout: 15_000 });
      await expect(runButton).toBeEnabled({ timeout: 15_000 });

      await page.getByLabel('返回数量').fill('1');
      const startResponsePromise = page.waitForResponse(
        (response) => response.url().includes('/api/v1/alphasift/screen/tasks')
          && response.request().method() === 'POST',
        { timeout: 15_000 },
      );
      await runButton.click();
      const startResponse = await startResponsePromise;
      expect(startResponse.status()).toBe(202);
      const accepted = await startResponse.json() as { task_id?: string; taskId?: string };
      const taskId = accepted.task_id || accepted.taskId || '';
      expect(taskId).toBeTruthy();

      await expect(page.getByText('选股任务运行中')).toBeVisible({ timeout: 15_000 });
      let latestScreenTask: RawAlphaSiftScreenTaskStatus = {};
      await expect.poll(async () => {
        latestScreenTask = await getAlphaSiftScreenTaskStatus(page.request, taskId);
        return latestScreenTask.status || 'unknown';
      }, {
        intervals: [2000, 3000, 5000],
        timeout: 180_000,
      }).toMatch(/^(completed|failed)$/);

      if (latestScreenTask.status === 'failed') {
        if (isDeterministicFakeAnalysisEnabled()) {
          throw new Error(
            `Deterministic AlphaSift smoke failed: ${latestScreenTask.error || latestScreenTask.message || 'unknown error'}`,
          );
        }
        await expect(page.getByText('调用失败')).toBeVisible({ timeout: 15_000 });
        await expect(page.getByText(/选股任务失败/)).toBeVisible({ timeout: 15_000 });
        await expect(page.getByText('0 条候选')).toBeVisible({ timeout: 15_000 });
      } else {
        const candidateCount = Number(
          latestScreenTask.result?.candidate_count
            ?? latestScreenTask.result?.candidateCount
            ?? latestScreenTask.result?.candidates?.length
            ?? 0,
        );
        await expect(page.getByText('选股完成')).toBeVisible({ timeout: 180_000 });
        await expect(page.getByText(`${candidateCount} 条候选`)).toBeVisible({ timeout: 15_000 });
        if (candidateCount > 0) {
          await expect.poll(async () => page.locator('tbody tr').count(), {
            timeout: 15_000,
          }).toBeGreaterThanOrEqual(1);
        } else {
          await expect(page.getByText('本次筛选没有返回候选，可调整策略参数或稍后重试。')).toBeVisible({ timeout: 15_000 });
        }
      }
    } else {
      await expect(page.getByText('AlphaSift 适配层不可用')).toBeVisible({ timeout: 15_000 });
      await expect(runButton).toBeDisabled();
    }

    expect(consoleWatch.errors).toEqual([]);
  } finally {
    await updateSystemConfigValue(page.request, 'ALPHASIFT_ENABLED', originalAlphaSiftEnabled).catch(() => undefined);
    pendingSystemConfigRestores.delete('ALPHASIFT_ENABLED');
    consoleWatch.stop();
  }
});

test('portfolio account, trade, cash ledger, corporate action, and FX refresh work from the browser', async ({ page, request }) => {
  const consoleWatch = watchConsole(page);
  const accountName = `${e2ePrefix}_portfolio`;

  try {
    await cleanupPortfolio(request);
    await openWorkbench(page, '/portfolio');

    if (await page.getByRole('button', { name: '新建账户' }).isVisible().catch(() => false)) {
      await page.getByRole('button', { name: '新建账户' }).click();
    }

    await page.getByPlaceholder('账户名称（必填）').fill(accountName);
    await page.getByPlaceholder('券商（可选，如 Demo/华泰）').fill('Codex');
    await page.getByRole('button', { name: '创建账户' }).click();
    await expect(page.getByRole('combobox').first()).toContainText(accountName, { timeout: 15_000 });

    await page.getByPlaceholder('股票代码（例如 600519）').fill('600519');
    await page.getByPlaceholder('数量（必填）').fill('1');
    await page.getByPlaceholder('成交价（必填）').fill('100');
    await page.getByRole('button', { name: '提交交易' }).click();
    await expect(page.getByText(/600519.*数量=1.*价格=100/)).toBeVisible({ timeout: 15_000 });

    await page.getByRole('button', { name: '删除' }).last().click();
    await page.getByRole('button', { name: '确认删除' }).click();
    await expect(page.getByText(/600519.*数量=1.*价格=100/)).not.toBeVisible({ timeout: 15_000 });

    const cashForm = page.locator('form', { has: page.getByRole('button', { name: '提交资金流水' }) });
    await cashForm.getByPlaceholder('金额').fill('1000');
    await cashForm.getByPlaceholder(/币种/).fill('CNY');
    await cashForm.getByRole('button', { name: '提交资金流水' }).click();
    const eventCard = page.getByRole('heading', { name: '事件记录' }).locator('..');
    await eventCard.locator('select').first().selectOption('cash');
    await eventCard.getByRole('button', { name: '刷新流水' }).click();
    await expect(eventCard.getByText(/流入 1000/)).toBeVisible({ timeout: 15_000 });
    await eventCard.getByRole('button', { name: '删除' }).click();
    await page.getByRole('button', { name: '确认删除' }).click();
    await expect(eventCard.getByText(/流入 1000/)).not.toBeVisible({ timeout: 15_000 });

    const corporateForm = page.locator('form', { has: page.getByRole('button', { name: '提交企业行为' }) });
    await corporateForm.getByPlaceholder('股票代码').fill('600519');
    await corporateForm.getByPlaceholder('每股分红').fill('0.1');
    await corporateForm.getByRole('button', { name: '提交企业行为' }).click();
    await eventCard.locator('select').first().selectOption('corporate');
    await eventCard.getByRole('button', { name: '刷新流水' }).click();
    await expect(eventCard.getByText(/现金分红 600519/)).toBeVisible({ timeout: 15_000 });
    await eventCard.getByRole('button', { name: '删除' }).click();
    await page.getByRole('button', { name: '确认删除' }).click();
    await expect(eventCard.getByText(/现金分红 600519/)).not.toBeVisible({ timeout: 15_000 });

    const fxResponsePromise = page.waitForResponse(
      (response) => response.url().includes('/api/v1/portfolio/fx/refresh') && response.request().method() === 'POST',
      { timeout: 15_000 },
    );
    await page.getByRole('button', { name: '刷新汇率' }).click();
    const fxResponse = await fxResponsePromise;
    expect(fxResponse.ok()).toBeTruthy();

    const activeAccountsBeforeDelete = await request.get('/api/v1/portfolio/accounts');
    expect(activeAccountsBeforeDelete.ok()).toBeTruthy();
    const activeAccountsBeforePayload = await activeAccountsBeforeDelete.json() as {
      accounts?: Array<{ id: number; name?: string; is_active?: boolean }>;
    };
    const createdAccount = (activeAccountsBeforePayload.accounts || []).find((account) => account.name === accountName);
    expect(createdAccount).toBeTruthy();

    const accountDeleteResponsePromise = page.waitForResponse(
      (response) => response.url().includes(`/api/v1/portfolio/accounts/${createdAccount!.id}`)
        && response.request().method() === 'DELETE',
      { timeout: 15_000 },
    );
    await page.getByRole('button', { name: '删除当前账户' }).click();
    const deleteDialog = page.getByRole('dialog', { name: '删除账户' });
    await expect(deleteDialog).toContainText(accountName);
    await expect(deleteDialog).toContainText('历史流水会保留');
    await deleteDialog.getByRole('button', { name: '确认删除' }).click();
    const accountDeleteResponse = await accountDeleteResponsePromise;
    expect(accountDeleteResponse.ok()).toBeTruthy();
    await expect(page.getByText(`账户「${accountName}」已从活跃视图移除。`)).toBeVisible({ timeout: 15_000 });

    const activeAccountsAfterDelete = await request.get('/api/v1/portfolio/accounts');
    expect(activeAccountsAfterDelete.ok()).toBeTruthy();
    const activeAccountsAfterPayload = await activeAccountsAfterDelete.json() as {
      accounts?: Array<{ id: number; name?: string; is_active?: boolean }>;
    };
    expect((activeAccountsAfterPayload.accounts || []).some((account) => account.name === accountName)).toBe(false);

    const inactiveAccountsAfterDelete = await request.get('/api/v1/portfolio/accounts', {
      params: { include_inactive: true },
    });
    expect(inactiveAccountsAfterDelete.ok()).toBeTruthy();
    const inactiveAccountsAfterPayload = await inactiveAccountsAfterDelete.json() as {
      accounts?: Array<{ id: number; name?: string; is_active?: boolean }>;
    };
    expect((inactiveAccountsAfterPayload.accounts || []).find((account) => account.name === accountName)).toMatchObject({
      id: createdAccount!.id,
      is_active: false,
    });

    expect(consoleWatch.errors).toEqual([]);
  } finally {
    consoleWatch.stop();
    await cleanupPortfolio(request);
  }
});

test('alerts rule can be created disabled and permanently deleted from the browser', async ({ page, request }) => {
  const consoleWatch = watchConsole(page);
  const ruleName = `${e2ePrefix}_alert`;
  const readRule = async () => {
    const response = await page.request.get('/api/v1/alerts/rules', { params: { page_size: 100 } });
    expect(response.ok()).toBeTruthy();
    const payload = await response.json() as { items?: Array<{ id: number; name?: string; enabled?: boolean }> };
    return (payload.items || []).find((rule) => rule.name === ruleName) || null;
  };

  try {
    await cleanupAlerts(request);
    await openWorkbench(page, '/alerts');

    await page.getByLabel('规则名称').fill(ruleName);
    await page.getByLabel('标的代码').fill('600519');
    await page.getByLabel('价格阈值').fill('9999');
    await page.getByLabel('创建后立即启用').uncheck();
    await page.getByRole('button', { name: '创建规则' }).click();

    await expect(page.getByText(`已创建告警规则「${ruleName}」`)).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText(ruleName, { exact: true })).toBeVisible({ timeout: 15_000 });
    const ruleRow = page.locator('tr', { hasText: ruleName });
    await expect(ruleRow.getByText('已停用')).toBeVisible();
    expect(await readRule()).toMatchObject({ name: ruleName, enabled: false });

    const enableResponsePromise = page.waitForResponse(
      (response) => response.url().includes('/api/v1/alerts/rules/')
        && response.url().includes('/enable')
        && response.request().method() === 'POST',
      { timeout: 15_000 },
    );
    await ruleRow.getByRole('button', { name: '启用' }).click();
    const enableResponse = await enableResponsePromise;
    expect(enableResponse.ok()).toBeTruthy();
    await expect(ruleRow.getByText('已启用')).toBeVisible({ timeout: 15_000 });
    expect(await readRule()).toMatchObject({ name: ruleName, enabled: true });

    const disableResponsePromise = page.waitForResponse(
      (response) => response.url().includes('/api/v1/alerts/rules/')
        && response.url().includes('/disable')
        && response.request().method() === 'POST',
      { timeout: 15_000 },
    );
    await ruleRow.getByRole('button', { name: '停用' }).click();
    const disableResponse = await disableResponsePromise;
    expect(disableResponse.ok()).toBeTruthy();
    await expect(ruleRow.getByText('已停用')).toBeVisible({ timeout: 15_000 });
    expect(await readRule()).toMatchObject({ name: ruleName, enabled: false });

    await page.getByRole('button', { name: `删除 ${ruleName}` }).click();
    await page.getByRole('button', { name: '删除', exact: true }).click();
    await expect(page.getByRole('table').getByText(ruleName, { exact: true })).not.toBeVisible({ timeout: 15_000 });

    expect(consoleWatch.errors).toEqual([]);
  } finally {
    consoleWatch.stop();
    await cleanupAlerts(request);
  }
});

test('chat, settings, and AlphaSift pages expose safe self-service states', async ({ page }) => {
  const consoleWatch = watchConsole(page);
  const probe = `${e2ePrefix}_chat_invalid_key_probe 600519`;

  try {
    await openWorkbench(page, '/chat');
    await expect(page.getByTestId('chat-workspace')).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText('策略', { exact: true })).toBeVisible({ timeout: 15_000 });
    await page.getByPlaceholder(/分析 600519/).fill(probe);
    await page.getByRole('button', { name: '发送', exact: true }).click();
    await expect(page.locator('p').filter({ hasText: probe }).last()).toBeVisible({ timeout: 15_000 });

    await openWorkbench(page, '/settings');
    await expect(page.getByRole('heading', { name: '系统设置' })).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText('AI 模型')).toBeVisible({ timeout: 15_000 });

    await openWorkbench(page, '/screening');
    await expect(page.getByRole('heading', { name: 'AlphaSift 选股' })).toBeVisible({ timeout: 15_000 });
    const runButton = page.getByRole('button', { name: '运行选股' });
    if (await page.getByText('AlphaSift 未开启').isVisible().catch(() => false)) {
      await expect(runButton).toBeDisabled();
      await expect(page.getByRole('button', { name: '开启 AlphaSift' })).toBeVisible();
    } else {
      await expect(runButton).toBeVisible();
    }

    expect(consoleWatch.errors).toEqual([]);
  } finally {
    consoleWatch.stop();
  }
});
