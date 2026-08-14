import type React from 'react';
import { useState } from 'react';
import { History, RefreshCw, Trash2 } from 'lucide-react';
import type { BacktestRunSummary } from '../../types/backtest';
import type { UiLanguage } from '../../i18n/uiText';
import { ConfirmDialog, EmptyState, StatusDot } from '../common';

interface Props {
  runs: BacktestRunSummary[];
  loadingRunId?: string | null;
  selectedRunId?: string | null;
  refreshing?: boolean;
  onSelect: (run: BacktestRunSummary) => void;
  onRefresh: () => void;
  onDelete: (run: BacktestRunSummary) => void;
  language: UiLanguage;
}

function pct(value: unknown): string {
  const n = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(n) ? `${n.toFixed(2)}%` : '--';
}

function returnTone(value: unknown): string {
  const n = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(n) || n === 0) return 'text-foreground';
  return n > 0 ? 'text-red-500' : 'text-emerald-600';
}

function fmtTime(value: string): string {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString(undefined, {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function statusTone(status: string): 'success' | 'danger' | 'neutral' {
  if (status === 'completed') return 'success';
  if (status === 'failed' || status === 'cancelled') return 'danger';
  return 'neutral';
}

const StrategyRunHistory: React.FC<Props> = ({
  runs,
  loadingRunId,
  selectedRunId,
  refreshing = false,
  onSelect,
  onRefresh,
  onDelete,
  language,
}) => {
  const zh = language === 'zh';
  const [confirmTarget, setConfirmTarget] = useState<BacktestRunSummary | null>(null);
  return (
    <section className="min-w-0 border border-border/70 bg-background/40">
      <header className="flex h-10 items-center justify-between border-b border-border/70 px-3">
        <div className="flex items-center gap-2 text-sm font-medium text-foreground">
          <History className="h-4 w-4 text-cyan" />
          {zh ? '历史运行' : 'Run history'}
        </div>
        <button
          type="button"
          onClick={onRefresh}
          disabled={refreshing}
          className="inline-flex h-7 items-center gap-1.5 rounded-md px-2 text-xs text-secondary-text hover:bg-accent hover:text-foreground disabled:opacity-50"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${refreshing ? 'animate-spin' : ''}`} />
          {zh ? '刷新' : 'Refresh'}
        </button>
      </header>
      {runs.length === 0 ? (
        <EmptyState
          title={zh ? '暂无回测记录' : 'No backtest runs yet'}
          description={zh ? '运行一次回测后，结果会自动保存在这里，方便日后比较。' : 'Run a backtest and its result is saved here for later comparison.'}
          className="rounded-none border-0 shadow-none"
        />
      ) : (
        <ul role="list" className="max-h-[420px] divide-y divide-border/50 overflow-y-auto">
          {runs.map((run) => {
            const metrics = run.metrics as Record<string, unknown>;
            const totalReturn = metrics?.totalReturnPct;
            const maxDrawdown = metrics?.maxDrawdownPct;
            const active = run.id === selectedRunId;
            const loading = run.id === loadingRunId;
            return (
              <li key={run.id}>
                <div
                  className={`flex items-stretch transition-colors hover:bg-accent/60 ${active ? 'bg-cyan/8' : ''}`}
                >
                  <button
                    type="button"
                    onClick={() => onSelect(run)}
                    disabled={loading}
                    className={`flex min-w-0 flex-1 flex-col gap-1 px-3 py-2 text-left disabled:opacity-60`}
                  >
                    <span className="flex items-center justify-between gap-2 text-xs">
                      <span className="flex min-w-0 items-center gap-1.5">
                        <StatusDot tone={statusTone(run.status)} aria-label={run.status} />
                        <span className="truncate text-foreground">{fmtTime(run.createdAt)}</span>
                      </span>
                      <span className={`shrink-0 font-mono font-medium ${returnTone(totalReturn)}`}>
                        {pct(totalReturn)}
                      </span>
                    </span>
                    <span className="flex items-center justify-between gap-2 text-[11px] text-secondary-text">
                      <span className="truncate">
                        {zh ? '回撤' : 'MaxDD'} {pct(maxDrawdown)}
                        {typeof metrics?.tradingDays === 'number' ? ` · ${metrics.tradingDays}${zh ? '天' : 'd'}` : ''}
                      </span>
                      <span className="truncate font-mono">
                        {typeof run.dataQuality?.latestTradeDate === 'string'
                          ? run.dataQuality.latestTradeDate
                          : (run.dataQuality?.datasetId as string | undefined) ?? '--'}
                      </span>
                    </span>
                  </button>
                  <button
                    type="button"
                    onClick={() => setConfirmTarget(run)}
                    disabled={loading}
                    className="flex w-9 shrink-0 items-center justify-center text-secondary-text transition-colors hover:bg-red-500/10 hover:text-red-500 disabled:opacity-60"
                    aria-label={zh ? `删除 ${fmtTime(run.createdAt)} 的运行记录` : `Delete run ${fmtTime(run.createdAt)}`}
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </div>
              </li>
            );
          })}
        </ul>
      )}
      <ConfirmDialog
        isOpen={confirmTarget !== null}
        isDanger
        title={zh ? '删除运行记录' : 'Delete run'}
        message={zh ? '确定删除这条回测记录？此操作不可恢复。' : 'Delete this backtest run? This cannot be undone.'}
        confirmText={zh ? '删除' : 'Delete'}
        onConfirm={() => {
          if (confirmTarget) onDelete(confirmTarget);
          setConfirmTarget(null);
        }}
        onCancel={() => setConfirmTarget(null)}
      />
    </section>
  );
};

export default StrategyRunHistory;
