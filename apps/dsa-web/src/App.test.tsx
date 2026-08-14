import { fireEvent, render, screen, within } from '@testing-library/react';
import type { ReactNode } from 'react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import App from './App';
import * as AuthContext from './contexts/AuthContext';
import { UI_LANGUAGE_STORAGE_KEY } from './utils/uiLanguage';

type AuthState = ReturnType<typeof AuthContext.useAuth>;

const { stockPageShouldThrow, setCurrentRoute, useAgentChatStoreMock } = vi.hoisted(() => {
  const setCurrentRoute = vi.fn();
  const stockPageShouldThrow = { value: false };
  const state = { completionBadge: false };
  const useAgentChatStoreMock = Object.assign(
    vi.fn((selector?: (value: typeof state) => unknown) => (selector ? selector(state) : state)),
    { getState: () => ({ setCurrentRoute }) },
  );
  return { stockPageShouldThrow, setCurrentRoute, useAgentChatStoreMock };
});

vi.mock('./contexts/AuthContext', () => ({
  AuthProvider: ({ children }: { children: ReactNode }) => children,
  useAuth: vi.fn(),
}));

vi.mock('./stores/agentChatStore', () => ({
  useAgentChatStore: useAgentChatStoreMock,
}));

vi.mock('./pages/HomePage', () => ({
  default: () => {
    if (stockPageShouldThrow.value) {
      throw new Error('chunk load failed');
    }
    return <div data-testid="home-page">Stock analysis</div>;
  },
}));

vi.mock('./pages/ResearchPage', () => ({
  default: () => <div data-testid="research-page">Research</div>,
}));

vi.mock('./pages/WorkflowPage', () => ({
  default: () => <div data-testid="workflow-page">Workflow</div>,
}));

vi.mock('./pages/DataPage', () => ({
  default: () => <div data-testid="data-page">Data</div>,
}));

vi.mock('./pages/OtherPage', () => ({
  default: () => <div data-testid="other-page">Other</div>,
}));

vi.mock('./pages/ChatPage', () => ({
  default: () => <div data-testid="chat-page">Chat</div>,
}));

vi.mock('./pages/PortfolioPage', () => ({
  default: () => <div data-testid="portfolio-page">Portfolio</div>,
}));

vi.mock('./pages/DecisionSignalsPage', () => ({
  default: () => <div data-testid="decision-signals-page">Decision signals</div>,
}));

vi.mock('./pages/BacktestPage', () => ({
  default: () => <div data-testid="backtest-page">Backtest</div>,
}));

vi.mock('./pages/EtfMomentumReportPage', () => ({
  default: () => <div data-testid="etf-momentum-report-page">ETF Momentum Report</div>,
}));

vi.mock('./pages/AlertsPage', () => ({
  default: () => <div data-testid="alerts-page">Alerts</div>,
}));

vi.mock('./pages/TokenUsagePage', () => ({
  default: () => <div data-testid="token-usage-page">Usage</div>,
}));

vi.mock('./pages/SettingsPage', () => ({
  default: () => <div data-testid="settings-page">Settings</div>,
}));

vi.mock('./pages/NotFoundPage', () => ({
  default: () => <div data-testid="not-found-page">Not Found</div>,
}));

vi.mock('./pages/LoginPage', () => ({
  default: () => <div data-testid="login-page">Login</div>,
}));

function makeAuthState(overrides: Partial<AuthState> = {}): AuthState {
  return {
    authEnabled: false,
    loggedIn: false,
    passwordSet: false,
    passwordChangeable: false,
    setupState: 'no_password',
    isLoading: false,
    loadError: null,
    login: vi.fn().mockResolvedValue({ success: true }),
    changePassword: vi.fn().mockResolvedValue({ success: true }),
    logout: vi.fn().mockResolvedValue(undefined),
    refreshStatus: vi.fn().mockResolvedValue(undefined),
    ...overrides,
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  stockPageShouldThrow.value = false;
  window.history.pushState({}, '', '/');
  localStorage.setItem(UI_LANGUAGE_STORAGE_KEY, 'zh');
  vi.mocked(AuthContext.useAuth).mockReturnValue(makeAuthState());
});

describe('App routing behavior', () => {
  it('renders the lightweight landing page at the root route', async () => {
    render(<App />);

    expect(await screen.findByTestId('landing-page')).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Daily Stock Analysis' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /九点猫研/ })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /MLflow/ })).toBeInTheDocument();

    const projectRelationship = screen.getByTestId('project-relationship');
    expect(within(projectRelationship).getByText('daily_stock_analysis')).toBeInTheDocument();
    expect(within(projectRelationship).getByText('quant_data')).toBeInTheDocument();
    expect(within(projectRelationship).getByText('quant_workflow + Qlib')).toBeInTheDocument();
    expect(within(projectRelationship).getByText('MLflow')).toBeInTheDocument();
    expect(within(projectRelationship).getByText('china-astock-quant')).toBeInTheDocument();
    expect(within(projectRelationship).getByText('独立数据项目与服务')).toBeInTheDocument();
    expect(within(projectRelationship).getByText('DSA 仓内量化研究执行层')).toBeInTheDocument();
    expect(within(projectRelationship).getByText('实验记录与产物存储')).toBeInTheDocument();
    expect(within(projectRelationship).getByText('独立可选策略参考库')).toBeInTheDocument();
    expect(within(projectRelationship).getByText('提供已验收数据快照')).toBeInTheDocument();
    expect(within(projectRelationship).getByText('记录 Run、参数、指标与产物')).toBeInTheDocument();
    expect(within(projectRelationship).getByText('DSA 聚合展示研究与结果')).toBeInTheDocument();
    expect(within(projectRelationship).getByText('可选参考')).toBeInTheDocument();
    expect(within(projectRelationship).queryByRole('link')).not.toBeInTheDocument();
    expect(screen.queryByTestId('home-page')).not.toBeInTheDocument();
  });

  it('shows loading fallback while auth status is initializing', () => {
    vi.mocked(AuthContext.useAuth).mockReturnValue(makeAuthState({ isLoading: true }));

    const { container } = render(<App />);

    expect(container.querySelector('.border-t-cyan')).toBeInTheDocument();
  });

  it('redirects protected routes to login when auth is enabled but user is not logged in', async () => {
    vi.mocked(AuthContext.useAuth).mockReturnValue(makeAuthState({
      authEnabled: true,
      loggedIn: false,
      setupState: 'enabled',
    }));
    window.history.pushState({}, '', '/portfolio');

    render(<App />);

    expect(await screen.findByTestId('login-page')).toBeInTheDocument();
    expect(window.location.pathname).toBe('/login');
    expect(window.location.search).toBe('?redirect=%2Fportfolio');
  });

  it('renders stock analysis on its dedicated route after auth is ready', async () => {
    window.history.pushState({}, '', '/stock-analysis');

    render(<App />);

    expect(await screen.findByTestId('home-page')).toBeInTheDocument();
    expect(setCurrentRoute).toHaveBeenCalledWith('/stock-analysis');
    expect(screen.queryByTestId('login-page')).not.toBeInTheDocument();
    expect(screen.queryByTestId('landing-page')).not.toBeInTheDocument();
  });

  it('renders the research workspace route after auth is ready', async () => {
    window.history.pushState({}, '', '/research');

    render(<App />);

    expect(await screen.findByTestId('research-page')).toBeInTheDocument();
    expect(setCurrentRoute).toHaveBeenCalledWith('/research');
  });

  it('renders the workflow route after auth is ready', async () => {
    window.history.pushState({}, '', '/workflow');

    render(<App />);

    expect(await screen.findByTestId('workflow-page')).toBeInTheDocument();
    expect(setCurrentRoute).toHaveBeenCalledWith('/workflow');
  });

  it('renders the separate data and other quant task routes', async () => {
    window.history.pushState({}, '', '/data');
    const { unmount } = render(<App />);
    expect(await screen.findByTestId('data-page')).toBeInTheDocument();
    expect(setCurrentRoute).toHaveBeenCalledWith('/data');

    unmount();
    window.history.pushState({}, '', '/other');
    render(<App />);
    expect(await screen.findByTestId('other-page')).toBeInTheDocument();
    expect(setCurrentRoute).toHaveBeenCalledWith('/other');
  });

  it('renders the dedicated ETF momentum report route', async () => {
    window.history.pushState({}, '', '/backtest/etf-momentum');

    render(<App />);

    expect(await screen.findByTestId('etf-momentum-report-page')).toBeInTheDocument();
    expect(setCurrentRoute).toHaveBeenCalledWith('/backtest/etf-momentum');
  });

  it.each(['/chat', '/decision-signals', '/usage'])('redirects retired route %s to the landing page', async (path) => {
    window.history.pushState({}, '', path);

    render(<App />);

    expect(await screen.findByTestId('landing-page')).toBeInTheDocument();
    expect(window.location.pathname).toBe('/');
    expect(screen.queryByTestId('chat-page')).not.toBeInTheDocument();
    expect(screen.queryByTestId('decision-signals-page')).not.toBeInTheDocument();
    expect(screen.queryByTestId('token-usage-page')).not.toBeInTheDocument();
  });

  it('redirects authenticated login visits back to the home page', async () => {
    vi.mocked(AuthContext.useAuth).mockReturnValue(makeAuthState({
      authEnabled: true,
      loggedIn: true,
      setupState: 'enabled',
    }));
    window.history.pushState({}, '', '/login');

    render(<App />);

    expect(await screen.findByTestId('landing-page')).toBeInTheDocument();
    expect(screen.queryByTestId('login-page')).not.toBeInTheDocument();
  });

  it('keeps the shell mounted and resets the route boundary after page render errors', async () => {
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    stockPageShouldThrow.value = true;
    window.history.pushState({}, '', '/stock-analysis');

    try {
      render(<App />);

      expect(await screen.findByRole('heading', { name: '页面加载失败' })).toBeInTheDocument();
      expect(screen.getByRole('navigation', { name: '主导航' })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: '重新加载页面' })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: '返回首页' })).toBeInTheDocument();

      stockPageShouldThrow.value = false;
      fireEvent.click(screen.getByRole('link', { name: '持仓' }));

      expect(await screen.findByTestId('portfolio-page')).toBeInTheDocument();
      expect(screen.queryByRole('heading', { name: '页面加载失败' })).not.toBeInTheDocument();
    } finally {
      consoleError.mockRestore();
    }
  });
});
