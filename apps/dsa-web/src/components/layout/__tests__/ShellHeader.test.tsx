import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { UiLanguageProvider } from '../../../contexts/UiLanguageContext';
import { UI_LANGUAGE_STORAGE_KEY } from '../../../utils/uiLanguage';
import { ThemeProvider } from '../../theme/ThemeProvider';
import { ShellHeader } from '../ShellHeader';

beforeAll(() => {
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: vi.fn().mockImplementation((query: string) => ({
      matches: query === '(prefers-color-scheme: dark)',
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    })),
  });
});

describe('ShellHeader', () => {
  beforeEach(() => {
    window.localStorage.setItem(UI_LANGUAGE_STORAGE_KEY, 'zh');
  });

  it('shows the workflow route title and description', () => {
    render(
      <MemoryRouter initialEntries={['/workflow']}>
        <UiLanguageProvider>
          <ThemeProvider>
            <ShellHeader
              collapsed={false}
              onToggleSidebar={vi.fn()}
              onOpenMobileNav={vi.fn()}
            />
          </ThemeProvider>
        </UiLanguageProvider>
      </MemoryRouter>,
    );

    expect(screen.getByText('工作流')).toBeInTheDocument();
    expect(screen.getByText('Qlib 转换、训练和回测任务中心')).toBeInTheDocument();
  });
});
