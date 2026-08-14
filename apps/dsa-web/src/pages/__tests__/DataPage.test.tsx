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
      validatedDatasets: 2,
      checkedAt: '2026-08-11T03:30:00Z',
      datasets: [
        {
          datasetId: 'ashare_daily_qfq',
          snapshotId: 'ashare-snapshot-1',
          manifestSha256: 'a'.repeat(64),
          contentSha256: 'b'.repeat(64),
          latestTradeDate: '2026-08-04',
          priceBasis: 'qfq',
          dataKind: 'daily_bars',
          role: 'canonical',
          status: 'ready',
        },
        {
          datasetId: 'cn_etf_daily',
          snapshotId: 'etf-snapshot-1',
          manifestSha256: 'c'.repeat(64),
          contentSha256: 'd'.repeat(64),
          latestTradeDate: '2026-08-07',
          priceBasis: 'qfq',
          dataKind: 'paired_daily_bars',
          role: 'canonical',
          status: 'ready',
        },
      ],
    });
  });

  it('renders only the governed quant_data readiness and immutable identities', async () => {
    render(
      <UiLanguageProvider>
        <DataPage />
      </UiLanguageProvider>,
    );

    expect(await screen.findByRole('heading', { name: '数据状态' })).toBeInTheDocument();
    expect(screen.getByText('2 个正式数据集')).toBeInTheDocument();
    expect(screen.getByText('A股前复权日线')).toBeInTheDocument();
    expect(screen.getByText('ETF 日线')).toBeInTheDocument();
    expect(screen.getByText('ashare-snapshot-1')).toBeInTheDocument();
    expect(screen.getByText('etf-snapshot-1')).toBeInTheDocument();
    expect(screen.queryByText('旧数据任务中心')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /下载|发布|构建/ })).not.toBeInTheDocument();
    await waitFor(() => expect(getDataStatus).toHaveBeenCalledTimes(1));
  });
});
