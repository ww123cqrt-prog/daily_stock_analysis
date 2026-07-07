import { fireEvent, render, screen } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { describe, expect, it } from 'vitest';
import NotFoundPage from '../NotFoundPage';

function renderNotFoundRoute() {
  return render(
    <MemoryRouter initialEntries={['/missing-page']}>
      <Routes>
        <Route path="/" element={<div>首页已打开</div>} />
        <Route path="*" element={<NotFoundPage />} />
      </Routes>
    </MemoryRouter>,
  );
}

describe('NotFoundPage', () => {
  it('renders a localized 404 state and updates the page title', () => {
    renderNotFoundRoute();

    expect(screen.getByText('404')).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: '页面未找到' })).toBeInTheDocument();
    expect(screen.getByText('抱歉，您访问的页面不存在或已被移动')).toBeInTheDocument();
    expect(document.title).toBe('页面未找到 - DSA');
  });

  it('navigates back to the home route', () => {
    renderNotFoundRoute();

    fireEvent.click(screen.getByRole('button', { name: '返回首页' }));

    expect(screen.getByText('首页已打开')).toBeInTheDocument();
  });
});
