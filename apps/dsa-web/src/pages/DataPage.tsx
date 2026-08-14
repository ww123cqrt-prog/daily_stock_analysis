import type React from 'react';
import { useCallback, useEffect, useState } from 'react';
import { Database, RefreshCw } from 'lucide-react';
import { getParsedApiError, type ParsedApiError } from '../api/error';
import {
  quantApi,
  type QuantDataStatus,
  type QuantDataStatusDataset,
} from '../api/quant';
import {
  ApiErrorAlert,
  AppPage,
  Badge,
  Button,
  Loading,
  PageHeader,
  StatusDot,
} from '../components/common';
import { useUiLanguage } from '../contexts/UiLanguageContext';

const DATASET_NAMES: Record<string, { zh: string; en: string }> = {
  ashare_daily_qfq: { zh: 'A股前复权日线', en: 'A-share adjusted daily bars' },
  ashare_daily_raw: { zh: 'A股原始日线', en: 'A-share raw daily bars' },
  ashare_fundamentals_pit: { zh: '防前视基本面', en: 'Point-in-time fundamentals' },
  cn_etf_daily: { zh: 'ETF 日线', en: 'ETF daily bars' },
  cn_index_daily: { zh: '指数日线', en: 'Index daily bars' },
  pit_csi800: { zh: 'CSI800 历史成分', en: 'CSI800 point-in-time members' },
  qlib_cleaning_plan: { zh: 'Qlib 清洗规则', en: 'Qlib cleaning plan' },
  robust_reference: { zh: '研究参考表', en: 'Research reference tables' },
};

function datasetName(dataset: QuantDataStatusDataset, language: 'zh' | 'en'): string {
  return DATASET_NAMES[dataset.datasetId]?.[language] ?? dataset.datasetId;
}

function formatCheckedAt(value: string, language: 'zh' | 'en'): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '-';
  return new Intl.DateTimeFormat(language === 'zh' ? 'zh-CN' : 'en-US', {
    dateStyle: 'medium',
    timeStyle: 'medium',
  }).format(date);
}

const DataPage: React.FC = () => {
  const { language } = useUiLanguage();
  const zh = language === 'zh';
  const [status, setStatus] = useState<QuantDataStatus | null>(null);
  const [error, setError] = useState<ParsedApiError | null>(null);
  const [loading, setLoading] = useState(true);

  const loadStatus = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setStatus(await quantApi.getDataStatus());
    } catch (nextError) {
      setError(getParsedApiError(nextError));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    document.title = `${zh ? '数据状态' : 'Data Status'} - DSA`;
    void loadStatus();
  }, [loadStatus, zh]);

  return (
    <AppPage className="max-w-[1400px] space-y-4">
      <PageHeader
        title={zh ? '数据状态' : 'Data Status'}
        description={zh ? 'quant_data 数据后端的只读状态' : 'Read-only status from the quant_data backend'}
        actions={(
          <Button
            variant="secondary"
            size="sm"
            onClick={() => void loadStatus()}
            disabled={loading}
            aria-label={zh ? '刷新数据状态' : 'Refresh data status'}
          >
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            {zh ? '刷新' : 'Refresh'}
          </Button>
        )}
      />

      {error ? (
        <ApiErrorAlert
          error={error}
          actionLabel={zh ? '重试' : 'Retry'}
          onAction={() => void loadStatus()}
        />
      ) : null}

      {loading && !status ? <Loading label={zh ? '正在读取数据状态' : 'Loading data status'} /> : null}

      {status ? (
        <>
          <section className="grid gap-px overflow-hidden rounded-lg border border-border/70 bg-border/70 sm:grid-cols-3">
            <div className="bg-card px-4 py-4">
              <p className="text-xs text-secondary-text">{zh ? '服务状态' : 'Service'}</p>
              <div className="mt-2 flex items-center gap-2 text-sm font-semibold text-foreground">
                <StatusDot tone={status.status === 'ready' ? 'success' : 'warning'} />
                {status.status === 'ready' ? 'Ready' : status.status}
              </div>
            </div>
            <div className="bg-card px-4 py-4">
              <p className="text-xs text-secondary-text">{zh ? '正式数据集' : 'Canonical datasets'}</p>
              <p className="mt-2 text-sm font-semibold text-foreground">
                {zh ? `${status.validatedDatasets} 个正式数据集` : `${status.validatedDatasets} datasets`}
              </p>
            </div>
            <div className="bg-card px-4 py-4">
              <p className="text-xs text-secondary-text">{zh ? '检查时间' : 'Checked at'}</p>
              <p className="mt-2 text-sm font-semibold text-foreground">{formatCheckedAt(status.checkedAt, language)}</p>
            </div>
          </section>

          <section className="overflow-hidden rounded-lg border border-border/70 bg-card" aria-label={zh ? '正式数据集列表' : 'Canonical dataset list'}>
            <div className="flex items-center gap-2 border-b border-border/70 px-4 py-3">
              <Database className="h-4 w-4 text-cyan" />
              <h2 className="text-sm font-semibold text-foreground">{zh ? '正式数据集' : 'Canonical datasets'}</h2>
            </div>
            <div className="divide-y divide-border/60">
              {status.datasets.map((dataset) => (
                <article
                  key={dataset.datasetId}
                  className="grid min-w-0 gap-4 px-4 py-4 lg:grid-cols-[minmax(180px,1.1fr)_120px_130px_minmax(260px,1.4fr)]"
                >
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <h3 className="text-sm font-semibold text-foreground">{datasetName(dataset, language)}</h3>
                      <Badge variant="success" size="sm">{dataset.status}</Badge>
                    </div>
                    <p className="mt-1 break-all font-mono text-xs text-secondary-text">{dataset.datasetId}</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-text">{zh ? '最新交易日' : 'Latest date'}</p>
                    <p className="mt-1 text-sm text-foreground">{dataset.latestTradeDate || '-'}</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-text">{zh ? '价格口径' : 'Price basis'}</p>
                    <p className="mt-1 text-sm text-foreground">{dataset.priceBasis || '-'}</p>
                  </div>
                  <div className="min-w-0">
                    <p className="text-xs text-muted-text">{zh ? '快照身份' : 'Snapshot identity'}</p>
                    <p className="mt-1 break-all font-mono text-xs text-foreground">{dataset.snapshotId}</p>
                    <p className="mt-2 break-all font-mono text-[11px] leading-4 text-muted-text" title={dataset.contentSha256}>
                      content {dataset.contentSha256}
                    </p>
                    <p className="mt-1 break-all font-mono text-[11px] leading-4 text-muted-text" title={dataset.manifestSha256}>
                      manifest {dataset.manifestSha256}
                    </p>
                  </div>
                </article>
              ))}
            </div>
          </section>
        </>
      ) : null}
    </AppPage>
  );
};

export default DataPage;
