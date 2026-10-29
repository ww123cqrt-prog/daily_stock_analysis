import type React from 'react';
import { useCallback, useEffect, useState } from 'react';
import { Database, RefreshCw } from 'lucide-react';
import { getParsedApiError, type ParsedApiError } from '../api/error';
import {
  quantApi,
  type QuantDataStatus,
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
        description={zh ? '冻结 QMT 实验输入，训练与回测共用同一份数据' : 'Frozen QMT input shared by training and backtesting'}
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
              <p className="text-xs text-secondary-text">{zh ? '实验输入状态' : 'Input status'}</p>
              <div className="mt-2 flex items-center gap-2 text-sm font-semibold">
                <StatusDot tone={status.status === 'ready' ? 'success' : 'warning'} />
                {status.status}
              </div>
            </div>
            <div className="bg-card px-4 py-4">
              <p className="text-xs text-secondary-text">{zh ? '所选范围' : 'Selected scope'}</p>
              <p className="mt-2 text-sm font-semibold">{status.experiment
                ? (zh ? `${status.experiment.symbolCount} 个标的 · ${status.experiment.calendarDays} 个交易日`
                  : `${status.experiment.symbolCount} symbols · ${status.experiment.calendarDays} trading days`)
                : '-'}</p>
            </div>
            <div className="bg-card px-4 py-4">
              <p className="text-xs text-secondary-text">{zh ? '检查时间' : 'Checked at'}</p>
              <p className="mt-2 text-sm font-semibold">{formatCheckedAt(status.checkedAt, language)}</p>
            </div>
          </section>
          <p className="text-xs text-secondary-text">
            {zh ? '完整冻结输入无需连接 QMT；状态可用不代表全市场历史齐备。缺失行情请在 QMT 中维护后重新导出。' : 'Complete frozen input works without QMT. Availability does not imply full market coverage. Maintain missing data in QMT before exporting again.'}
          </p>
          {status.summary ? <p className="text-sm text-secondary-text">{status.summary}</p> : null}
          {status.experiment ? (
            <section className="overflow-hidden rounded-lg border border-border/70 bg-card" aria-label={zh ? '冻结实验输入' : 'Frozen experiment input'}>
              <div className="flex items-center gap-2 border-b border-border/70 px-4 py-3">
                <Database className="h-4 w-4 text-cyan" />
                <h2 className="text-sm font-semibold text-foreground">{zh ? '冻结实验输入' : 'Frozen experiment input'}</h2>
                <Badge variant="success" size="sm">{zh ? '离线可读' : 'Offline ready'}</Badge>
              </div>
              <dl className="grid gap-4 px-4 py-4 sm:grid-cols-2">
                <div className="min-w-0">
                  <dt className="text-xs text-muted-text">{zh ? '实验 ID' : 'Experiment ID'}</dt>
                  <dd className="mt-1 break-all font-mono text-sm">{status.experiment.experimentId}</dd>
                </div>
                <div>
                  <dt className="text-xs text-muted-text">{zh ? '实际行情范围' : 'Observed bar range'}</dt>
                  <dd className="mt-1 text-sm">{status.experiment.actualStart} ~ {status.experiment.actualEnd}</dd>
                </div>
                <div>
                  <dt className="text-xs text-muted-text">{zh ? '价格与成交量口径' : 'Price and volume basis'}</dt>
                  <dd className="mt-1 text-sm">{status.experiment.priceBasis} · {[...new Set(Object.values(status.experiment.volumeUnits || {}))].join(' / ')}</dd>
                </div>
                <div>
                  <dt className="text-xs text-muted-text">{zh ? '标的列表性质' : 'Universe type'}</dt>
                  <dd className="mt-1 text-sm">{status.experiment.universeKind === 'current'
                    ? (zh ? '当前列表，包含存续偏差' : 'Current list; survivorship bias applies')
                    : status.experiment.universeKind === 'historical'
                      ? (zh ? '历史成员区间' : 'Historical membership')
                      : (zh ? '自定义列表' : 'Custom list')}</dd>
                </div>
                <div className="min-w-0 sm:col-span-2">
                  <dt className="text-xs text-muted-text">{zh ? '行情文件 SHA256' : 'Bars SHA256'}</dt>
                  <dd className="mt-1 break-all font-mono text-xs">{status.experiment.barsSha256}</dd>
                </div>
                {status.experiment.missingSymbols?.length ? (
                  <div className="sm:col-span-2 text-sm text-warning">
                    {zh ? '未返回行情的标的：' : 'Symbols without bars: '}{status.experiment.missingSymbols.join(', ')}
                  </div>
                ) : null}
              </dl>
            </section>
          ) : null}
        </>
      ) : null}
    </AppPage>
  );
};

export default DataPage;
