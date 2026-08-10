import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it, vi } from 'vitest';
import { SidebarNav } from '../SidebarNav';

const mockLogout = vi.fn().mockResolvedValue(undefined);
const mockGetScreeningStatus = vi.fn().mockResolvedValue({ enabled: false, available: false });
const mockThemeToggle = vi.fn(({ collapsed }: { collapsed?: boolean }) => (
  <button type="button">{collapsed ? '切换主题(折叠)' : '切换主题'}</button>
));

const completionBadgeState = { value: true };

vi.mock('../../../contexts/AuthContext', () => ({
  useAuth: () => ({
    authEnabled: true,
    logout: mockLogout,
  }),
}));

vi.mock('../../../stores/agentChatStore', () => ({
  useAgentChatStore: (selector: (state: { completionBadge: boolean }) => unknown) =>
    selector({ completionBadge: completionBadgeState.value }),
}));

vi.mock('../../../api/screening', () => ({
  SCREENING_CONFIG_CHANGED_EVENT: 'screening-config-changed',
  SYSTEM_CONFIG_CHANGED_EVENT: 'dsa-system-config-changed',
  screeningApi: {
    getStatus: () => mockGetScreeningStatus(),
  },
}));

vi.mock('../../theme/ThemeToggle', () => ({
  ThemeToggle: (props: { collapsed?: boolean }) => mockThemeToggle(props),
}));

describe('SidebarNav', () => {
  it('hides the screening navigation item while Screening is disabled', () => {
    mockGetScreeningStatus.mockResolvedValueOnce({ enabled: false, available: true });

    render(
      <MemoryRouter initialEntries={['/']}>
        <SidebarNav />
      </MemoryRouter>,
    );

    expect(screen.queryByRole('link', { name: '选股' })).not.toBeInTheDocument();
  });

  it('shows the research navigation item as a first-class entry', () => {
    render(
      <MemoryRouter initialEntries={['/research']}>
        <SidebarNav />
      </MemoryRouter>,
    );

    const researchLink = screen.getByRole('link', { name: '研究' });
    expect(researchLink).toHaveAttribute('href', '/research');
    expect(researchLink).toHaveClass('font-medium');
  });

  it('shows the workflow navigation item for browser-run quant tasks', () => {
    render(
      <MemoryRouter initialEntries={['/workflow']}>
        <SidebarNav />
      </MemoryRouter>,
    );

    const workflowLink = screen.getByRole('link', { name: '工作流' });
    expect(workflowLink).toHaveAttribute('href', '/workflow');
    expect(workflowLink).toHaveClass('font-medium');
  });

  it('opens the MLflow bilingual UI with a full-page navigation', () => {
    render(
      <MemoryRouter initialEntries={['/research']}>
        <SidebarNav />
      </MemoryRouter>,
    );

    expect(screen.getByRole('link', { name: '实验' })).toHaveAttribute('href', '/experiments/');
  });

  it('marks the experiments entry active without marking the language control active', () => {
    render(
      <MemoryRouter initialEntries={['/experiments/']}>
        <SidebarNav />
      </MemoryRouter>,
    );

    const experimentsLink = screen.getByRole('link', { name: '实验' });
    expect(experimentsLink).toHaveAttribute('aria-current', 'page');
    expect(experimentsLink).toHaveClass('font-medium');

    const languageButton = screen.getByRole('button', { name: '切换界面语言' });
    expect(languageButton).not.toHaveClass('font-medium');
    expect(languageButton).toHaveClass('h-[var(--nav-item-height)]', 'w-full');
  });

  it('passes the same stable dimensions to the theme control as navigation links', () => {
    render(
      <MemoryRouter initialEntries={['/workflow']}>
        <SidebarNav />
      </MemoryRouter>,
    );

    expect(screen.getByRole('link', { name: '工作流' })).toHaveClass('h-[var(--nav-item-height)]', 'w-full');
    expect(mockThemeToggle).toHaveBeenCalledWith(expect.objectContaining({
      triggerClassName: expect.stringContaining('h-[var(--nav-item-height)]'),
    }));
    expect(mockThemeToggle).toHaveBeenCalledWith(expect.objectContaining({
      triggerClassName: expect.stringContaining('w-full'),
    }));
  });

  it('shows screening directly after chat when Screening is enabled', async () => {
    mockGetScreeningStatus.mockResolvedValueOnce({ enabled: true, available: true });

    render(
      <MemoryRouter initialEntries={['/']}>
        <SidebarNav />
      </MemoryRouter>,
    );

    expect(await screen.findByRole('link', { name: '选股' })).toHaveAttribute('href', '/screening');
    const hrefs = screen.getAllByRole('link').map((link) => link.getAttribute('href'));
    expect(hrefs.slice(0, 6)).toEqual(['/', '/research', '/workflow', '/chat', '/screening', '/portfolio']);
  });

  it('refreshes the controlled screening entry after config changes', async () => {
    mockGetScreeningStatus
      .mockResolvedValueOnce({ enabled: false, available: true })
      .mockResolvedValueOnce({ enabled: true, available: true });

    render(
      <MemoryRouter initialEntries={['/']}>
        <SidebarNav />
      </MemoryRouter>,
    );

    expect(screen.queryByRole('link', { name: '选股' })).not.toBeInTheDocument();
    window.dispatchEvent(new Event('screening-config-changed'));

    expect(await screen.findByRole('link', { name: '选股' })).toHaveAttribute('href', '/screening');
    await waitFor(() => expect(mockGetScreeningStatus.mock.calls.length).toBeGreaterThanOrEqual(2));
  });

  it('shows the shared completion badge only when chat completion is pending', () => {
    completionBadgeState.value = true;

    const { rerender } = render(
      <MemoryRouter initialEntries={['/chat']}>
        <SidebarNav />
      </MemoryRouter>,
    );

    expect(screen.getByTestId('chat-completion-badge')).toBeInTheDocument();
    expect(screen.getByLabelText('问股有新消息')).toBeInTheDocument();

    completionBadgeState.value = false;
    rerender(
      <MemoryRouter initialEntries={['/chat']}>
        <SidebarNav />
      </MemoryRouter>,
    );

    expect(screen.queryByTestId('chat-completion-badge')).not.toBeInTheDocument();
  });

  it('renders the collapsed theme toggle variant when the sidebar is collapsed', () => {
    render(
      <MemoryRouter initialEntries={['/']}>
        <SidebarNav collapsed />
      </MemoryRouter>,
    );

    expect(mockThemeToggle).toHaveBeenCalledWith(
      expect.objectContaining({ variant: 'nav', collapsed: true }),
    );
    expect(screen.getByRole('button', { name: '切换主题(折叠)' })).toBeInTheDocument();
  });

  it('renders the alerts navigation item and marks it active', () => {
    render(
      <MemoryRouter initialEntries={['/alerts']}>
        <SidebarNav />
      </MemoryRouter>,
    );

    const alertsLink = screen.getByRole('link', { name: '告警' });
    expect(alertsLink).toHaveAttribute('href', '/alerts');
    expect(alertsLink).toHaveClass('font-medium');
  });

  it('renders the AI signals navigation item and marks it active', () => {
    render(
      <MemoryRouter initialEntries={['/decision-signals']}>
        <SidebarNav />
      </MemoryRouter>,
    );

    const signalsLink = screen.getByRole('link', { name: 'AI 建议' });
    expect(signalsLink).toHaveAttribute('href', '/decision-signals');
    expect(signalsLink).toHaveClass('font-medium');
  });

  it('opens the logout confirmation and confirms logout', async () => {
    render(
      <MemoryRouter initialEntries={['/chat']}>
        <SidebarNav />
      </MemoryRouter>,
    );

    fireEvent.click(screen.getByRole('button', { name: '退出' }));

    expect(await screen.findByRole('heading', { name: '退出登录' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: '确认退出' }));
    expect(mockLogout).toHaveBeenCalled();
  });
});
