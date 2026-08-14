import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import LandingPage from '../LandingPage';

const { getLocalServiceStatus, startLocalService } = vi.hoisted(() => ({
  getLocalServiceStatus: vi.fn(),
  startLocalService: vi.fn(),
}));

vi.mock('../../api/quant', () => ({
  quantApi: {
    getLocalServiceStatus: (service: string) => getLocalServiceStatus(service),
    startLocalService: (service: string) => startLocalService(service),
  },
}));

function renderLandingPage() {
  return render(
    <MemoryRouter>
      <LandingPage />
    </MemoryRouter>,
  );
}

describe('LandingPage local service links', () => {
  beforeEach(() => {
    getLocalServiceStatus.mockReset();
    startLocalService.mockReset();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('shows starting for CatDesk and opens it only after the service becomes ready', async () => {
    const open = vi.spyOn(window, 'open').mockImplementation(() => null);
    getLocalServiceStatus
      .mockResolvedValueOnce({ status: 'unavailable', openUrl: 'http://127.0.0.1:8765/' })
      .mockResolvedValueOnce({ status: 'available', openUrl: 'http://127.0.0.1:8765/' });
    startLocalService.mockResolvedValueOnce({ status: 'starting', openUrl: 'http://127.0.0.1:8765/' });

    renderLandingPage();
    fireEvent.click(screen.getByRole('button', { name: /九点猫研/ }));

    expect(await screen.findByText('正在启动')).toBeInTheDocument();
    await waitFor(() => {
      expect(open).toHaveBeenCalledWith('http://127.0.0.1:8765/', '_self');
    });
    expect(startLocalService).toHaveBeenCalledWith('catdesk');
  });

  it('opens an already available MLflow service without starting a duplicate process', async () => {
    const open = vi.spyOn(window, 'open').mockImplementation(() => null);
    getLocalServiceStatus.mockResolvedValueOnce({ status: 'available', openUrl: '/experiments/' });

    renderLandingPage();
    fireEvent.click(screen.getByRole('button', { name: /MLflow/ }));

    await waitFor(() => {
      expect(open).toHaveBeenCalledWith('/experiments/', '_self');
    });
    expect(startLocalService).not.toHaveBeenCalled();
  });
});
