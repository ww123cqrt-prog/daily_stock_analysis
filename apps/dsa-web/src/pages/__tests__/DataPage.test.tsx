import { render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { UiLanguageProvider } from '../../contexts/UiLanguageContext';
import { UI_LANGUAGE_STORAGE_KEY } from '../../utils/uiLanguage';
import DataPage from '../DataPage';

const { getDataStatus } = vi.hoisted(() => ({ getDataStatus: vi.fn() }));

vi.mock('../../api/quant', () => ({
  quantApi: { getDataStatus },
}));

vi.mock('../WorkflowPage', () => ({
  QuantTaskPage: () => <div>旧数据任务中心</div>,
}));

describe('DataPage', () => {
  beforeEach(() => {
    localStorage.clear();
    localStorage.setItem(UI_LANGUAGE_STORAGE_KEY, 'zh');
    getDataStatus.mockReset();
    getDataStatus.mockResolvedValue({
      status: 'ready',
      availableDatasets: 1,
      backend: "qmt_native",
      validationScope: "frozen_experiment",
      checkedAt: '2026-08-11T03:30:00Z',
      datasets: [],
      sdk: { status: 'not_checked', requiredForOffline: false },
      experiment: { experimentId: 'qmt-input-one', barsSha256: 'a'.repeat(64),
        priceBasis: 'front', actualStart: '2026-06-01', actualEnd: '2026-08-07',
        calendarStart: '2026-06-01', calendarEnd: '2026-08-07', calendarDays: 50,
        universeKind: 'current', symbolCount: 2, rows: 100, missingSymbols: [],
        volumeUnits: { SH600519: 'shares', SH510300: 'units' } },
    });
  });

  it('renders frozen experiment coverage independently of live QMT', async () => {
    render(
      <UiLanguageProvider>
        <DataPage />
      </UiLanguageProvider>,
    );

    expect(await screen.findByRole('heading', { name: '数据状态' })).toBeInTheDocument();
    expect(screen.getByText('qmt-input-one')).toBeInTheDocument();
    expect(screen.getByText(/2 个标的/)).toBeInTheDocument();
    expect(screen.getByText(/当前列表/)).toBeInTheDocument();
    expect(screen.getByText(/无需连接 QMT/)).toBeInTheDocument();
    expect(screen.getByText('a'.repeat(64))).toBeInTheDocument();
    expect(screen.queryByText('旧数据任务中心')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /下载|发布|构建/ })).not.toBeInTheDocument();
    await waitFor(() => expect(getDataStatus).toHaveBeenCalledTimes(1));
  });
});
