import type React from 'react';
import { AlertTriangle, CheckCircle2 } from 'lucide-react';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip as ChartTooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { UiLanguage } from '../../i18n/uiText';
import type { StrategyBacktestResponse } from '../../types/backtest';

interface Props {
  language: UiLanguage;
  report: StrategyBacktestResponse | null;
  emptyTitle: string;
}

function pct(value: number | null | undefined, digits = 2): string {
  return value == null || !Number.isFinite(value) ? '--' : `${value.toFixed(digits)}%`;
}

function number(value: number | null | undefined, digits = 2): string {
  return value == null || !Number.isFinite(value) ? '--' : value.toFixed(digits);
}

function money(value: number, language: UiLanguage): string {
  return new Intl.NumberFormat(language === 'zh' ? 'zh-CN' : 'en-US', {
    maximumFractionDigits: 2,
  }).format(value);
}

function tone(value: number | null | undefined): string {
  if (value == null || value === 0) return 'text-foreground';
  return value > 0 ? 'text-red-500' : 'text-emerald-600';
}

function recordText(record: Record<string, unknown>, key: string): string {
  const value = record[key];
  return value == null || value === '' ? '--' : String(value);
}

const INDEX_LABELS: Record<string, string> = {
  SH000300: '沪深300',
  SH000905: '中证500',
  SH000906: '中证800',
  SZ399006: '创业板指',
  SH000688: '科创50',
};

const POOL_LABELS: Record<string, { zh: string; en: string }> = {
  etf: { zh: 'ETF 池', en: 'ETF pool' },
  csi800: { zh: 'CSI800', en: 'CSI800' },
  all_a: { zh: '全 A', en: 'All A-shares' },
  custom: { zh: '自选', en: 'Custom' },
};

function comparisonLabel(symbol: string, name?: string): string {
  return name ?? INDEX_LABELS[symbol] ?? symbol.replace(/^strategy:/, '');
}

const StrategyBacktestResults: React.FC<Props> = ({ language, report, emptyTitle }) => {
  const zh = language === 'zh';
  if (!report) {
    return (
      <div className="flex min-h-[32rem] items-center justify-center border border-dashed border-border/80 bg-card/20 px-6 text-center">
        <div>
          <p className="text-base font-medium text-foreground">{emptyTitle}</p>
          <p className="mt-2 text-sm text-secondary-text">
            {zh ? '运行后在这里显示收益、风险、净值和交易记录。' : 'Returns, risk, equity and trades appear here after a run.'}
          </p>
        </div>
      </div>
    );
  }

  const metrics = report.metrics;
  const primaryMetrics = zh
    ? [
        ['交易策略总收益', pct(metrics.totalReturnPct), tone(metrics.totalReturnPct)],
        ['年化收益', pct(metrics.annualizedReturnPct), tone(metrics.annualizedReturnPct)],
        ['最大回撤', pct(metrics.maxDrawdownPct), tone(metrics.maxDrawdownPct)],
        ['Sharpe', number(metrics.sharpeRatio), tone(metrics.sharpeRatio)],
      ]
    : [
        ['Total return', pct(metrics.totalReturnPct), tone(metrics.totalReturnPct)],
        ['Annualized return', pct(metrics.annualizedReturnPct), tone(metrics.annualizedReturnPct)],
        ['Max drawdown', pct(metrics.maxDrawdownPct), tone(metrics.maxDrawdownPct)],
        ['Sharpe', number(metrics.sharpeRatio), tone(metrics.sharpeRatio)],
      ];
  const secondaryMetrics = zh
    ? [
        ['基准总收益', pct(metrics.benchmarkReturnPct), tone(metrics.benchmarkReturnPct)],
        ['基准年化', pct(metrics.benchmarkAnnualizedReturnPct), tone(metrics.benchmarkAnnualizedReturnPct)],
        ['基准最大回撤', pct(metrics.benchmarkMaxDrawdownPct), tone(metrics.benchmarkMaxDrawdownPct)],
        ['基准Sharpe', number(metrics.benchmarkSharpeRatio), tone(metrics.benchmarkSharpeRatio)],
        ['基准年化波动', pct(metrics.benchmarkAnnualizedVolatilityPct), 'text-foreground'],
        ['Alpha（年化）', pct(metrics.alphaPct), tone(metrics.alphaPct)],
        ['Beta', number(metrics.beta), 'text-foreground'],
        ['策略年化波动', pct(metrics.annualizedVolatilityPct), 'text-foreground'],
        ['日胜率', pct(metrics.winRatePct), tone(metrics.winRatePct - 50)],
        ['交易次数', String(metrics.tradeCount), 'text-foreground'],
      ]
    : [
        ['Benchmark return', pct(metrics.benchmarkReturnPct), tone(metrics.benchmarkReturnPct)],
        ['Benchmark annualized', pct(metrics.benchmarkAnnualizedReturnPct), tone(metrics.benchmarkAnnualizedReturnPct)],
        ['Benchmark max drawdown', pct(metrics.benchmarkMaxDrawdownPct), tone(metrics.benchmarkMaxDrawdownPct)],
        ['Benchmark Sharpe', number(metrics.benchmarkSharpeRatio), tone(metrics.benchmarkSharpeRatio)],
        ['Benchmark volatility', pct(metrics.benchmarkAnnualizedVolatilityPct), 'text-foreground'],
        ['Annualized alpha', pct(metrics.alphaPct), tone(metrics.alphaPct)],
        ['Beta', number(metrics.beta), 'text-foreground'],
        ['Strategy volatility', pct(metrics.annualizedVolatilityPct), 'text-foreground'],
        ['Daily win rate', pct(metrics.winRatePct), tone(metrics.winRatePct - 50)],
        ['Trades', String(metrics.tradeCount), 'text-foreground'],
      ];
  const comparisons = report.comparisons ?? {};
  const comparisonEntries = Object.entries(comparisons);
  const compareColors = ['#0ea5e9', '#f59e0b', '#10b981', '#a855f7', '#ec4899', '#84cc16', '#f97316', '#14b8a6'];
  const chartData = report.equityCurve.map((point, index) => {
    const row: Record<string, unknown> = { ...point };
    for (const [symbol, comparison] of comparisonEntries) {
      const equity = comparison.equity?.[index];
      row[`cmp_${symbol}`] = equity == null || !Number.isFinite(equity)
        ? null
        : (equity / metrics.initialCapital - 1) * 100;
    }
    return row;
  });

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border/70 pb-3">
        <div>
          <div className="flex items-center gap-2">
            <CheckCircle2 className="h-4 w-4 text-cyan" />
            <h2 className="text-base font-semibold text-foreground">
              {zh ? '回测结果' : 'Backtest result'}
            </h2>
          </div>
          <p className="mt-1 text-xs text-secondary-text">
            {(() => {
              const poolKey = String(report.dataQuality.pool ?? '');
              const poolLabel = POOL_LABELS[poolKey] ? (zh ? POOL_LABELS[poolKey].zh : POOL_LABELS[poolKey].en) : null;
              const poolText = poolLabel
                ? `${poolLabel}（${report.dataQuality.symbolCount ?? report.symbols.length} 只）`
                : report.symbols.join(' / ');
              return `${poolText} · ${report.period ? `${report.period.startDate} - ${report.period.endDate}` : '--'}`;
            })()}
          </p>
        </div>
        <span className="rounded-md border border-border/70 bg-card/60 px-2 py-1 font-mono text-xs text-secondary-text">
          {report.status}
        </span>
      </div>

      <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        {primaryMetrics.map(([label, value, className]) => (
          <div key={label} className="rounded-lg border border-border/70 bg-card px-4 py-3">
            <p className="text-xs text-secondary-text">{label}</p>
            <p className={`mt-1.5 text-2xl font-semibold tabular-nums leading-none ${className}`}>{value}</p>
          </div>
        ))}
      </section>

      <section className="grid grid-cols-3 gap-px overflow-hidden rounded-lg border border-border/70 bg-border/70 sm:grid-cols-6">
        {secondaryMetrics.map(([label, value, className]) => (
          <div key={label} className="bg-card px-3 py-2.5">
            <p className="text-[11px] text-secondary-text">{label}</p>
            <p className={`mt-1 text-sm font-medium tabular-nums ${className}`}>{value}</p>
          </div>
        ))}
      </section>

      {comparisonEntries.length ? (
        <section className="border border-border/70 bg-card/35 p-3">
          <h3 className="mb-3 text-sm font-medium text-foreground">
            {zh ? '对比基准' : 'Comparison benchmarks'}
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead className="text-left text-secondary-text">
                <tr>
                  <th className="py-1 pr-3 font-medium">{zh ? '基准' : 'Benchmark'}</th>
                  <th className="py-1 pr-3 text-right font-medium">{zh ? '区间收益' : 'Return'}</th>
                  <th className="py-1 pr-3 text-right font-medium">{zh ? '策略超额' : 'Excess'}</th>
                  <th className="py-1 text-right font-medium">{zh ? '最大回撤' : 'Max drawdown'}</th>
                </tr>
              </thead>
              <tbody>
                {comparisonEntries.map(([symbol, comparison]) => {
                  const excess = metrics.totalReturnPct - comparison.totalReturnPct;
                  return (
                    <tr key={symbol} className="border-t border-border/50">
                      <td className="py-2 pr-3 font-mono text-foreground">{comparisonLabel(symbol, comparison.name)}</td>
                      <td className={`py-2 pr-3 text-right font-mono tabular-nums ${tone(comparison.totalReturnPct)}`}>
                        {pct(comparison.totalReturnPct)}
                      </td>
                      <td className={`py-2 pr-3 text-right font-mono tabular-nums ${tone(excess)}`}>
                        {excess > 0 ? '+' : ''}{pct(excess)}
                      </td>
                      <td className="py-2 text-right font-mono tabular-nums text-secondary-text">
                        {pct(comparison.maxDrawdownPct)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      {report.equityCurve.length ? (
        <section className="border border-border/70 bg-card/35 p-3">
          <h3 className="mb-3 text-sm font-medium text-foreground">
            {zh ? '交易策略与对比基准净值' : 'Trading strategy and comparison equity'}
          </h3>
          <div className="h-72 min-w-0" data-testid="strategy-equity-chart">
            <ResponsiveContainer width="100%" height="100%" initialDimension={{ width: 640, height: 288 }}>
              <LineChart data={chartData} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.18)" />
                <XAxis dataKey="date" minTickGap={42} tick={{ fontSize: 10 }} />
                <YAxis tickFormatter={(value: number) => `${value.toFixed(0)}%`} tick={{ fontSize: 10 }} />
                <ChartTooltip formatter={value => `${Number(value).toFixed(2)}%`} />
                <Legend />
                <Line type="monotone" dataKey="strategyReturnPct" name={zh ? '交易策略' : 'Trading strategy'} stroke="#dc2626" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="benchmarkReturnPct" name={zh ? '基准' : 'Benchmark'} stroke="#2563eb" strokeWidth={1.8} dot={false} />
                {comparisonEntries.map(([symbol, comparison], index) => (
                  <Line
                    key={symbol}
                    type="monotone"
                    dataKey={`cmp_${symbol}`}
                    name={comparisonLabel(symbol, comparison.name)}
                    stroke={compareColors[index % compareColors.length]}
                    strokeWidth={1.5}
                    dot={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </section>
      ) : null}

      <section className="grid gap-3 border-y border-border/70 py-3 text-xs sm:grid-cols-2 xl:grid-cols-4">
        <div><span className="text-secondary-text">{zh ? '数据集' : 'Dataset'}</span><p className="mt-1 break-all font-mono text-foreground">{recordText(report.dataQuality, 'datasetId')}</p></div>
        <div><span className="text-secondary-text">{zh ? '数据快照' : 'Snapshot'}</span><p className="mt-1 break-all font-mono text-foreground">{recordText(report.dataQuality, 'snapshotId')}</p></div>
        <div><span className="text-secondary-text">{zh ? '价格口径' : 'Price basis'}</span><p className="mt-1 font-mono text-foreground">{recordText(report.dataQuality, 'priceBasis')}</p></div>
        <div><span className="text-secondary-text">{zh ? '数据来源' : 'Provider'}</span><p className="mt-1 font-mono text-foreground">{recordText(report.dataQuality, 'provider')}</p></div>
      </section>

      {report.warnings.length ? (
        <section className="space-y-2">
          {report.warnings.map(warning => (
            <div key={warning} className="flex items-start gap-2 border border-amber-500/25 bg-amber-500/8 px-3 py-2 text-xs text-amber-700">
              <AlertTriangle className="mt-0.5 h-4 w-4 flex-none" />
              <span>{warning}</span>
            </div>
          ))}
        </section>
      ) : null}

      {(report.comparisonFailures ?? []).length ? (
        <section className="space-y-2">
          {(report.comparisonFailures ?? []).map(failure => (
            <div key={failure.name} className="flex items-start gap-2 border border-rose-500/25 bg-rose-500/8 px-3 py-2 text-xs text-rose-600">
              <AlertTriangle className="mt-0.5 h-4 w-4 flex-none" />
              <span>
                {zh ? `比较策略「${failure.name}」未产出结果` : `Comparison strategy "${failure.name}" produced no result`}
                {failure.error ? `：${failure.error}` : ''}
              </span>
            </div>
          ))}
        </section>
      ) : null}

      {report.logs.length || Object.keys(report.scriptCheck).length ? (
        <section className="border border-border/70 bg-[#10151c] text-slate-200">
          <div className="flex items-center justify-between border-b border-white/10 px-3 py-2">
            <h3 className="text-xs font-medium">{zh ? '运行日志' : 'Run log'}</h3>
            <span className="font-mono text-[11px] text-slate-400">{recordText(report.scriptCheck, 'status')}</span>
          </div>
          <pre className="max-h-40 overflow-auto whitespace-pre-wrap px-3 py-3 font-mono text-xs leading-5 text-slate-300">
            {report.logs.length ? report.logs.join('\n') : (zh ? '静态安全检查通过；脚本没有输出日志。' : 'Static safety check passed; the script emitted no logs.')}
          </pre>
        </section>
      ) : null}

      <section>
        <div className="mb-2 flex items-center justify-between">
          <h3 className="text-sm font-medium text-foreground">{zh ? '成交记录' : 'Trades'}</h3>
          <span className="text-xs text-secondary-text">{zh ? `累计费用 ${money(metrics.totalCost, language)}` : `Total fees ${money(metrics.totalCost, language)}`}</span>
        </div>
        {report.trades.length ? (
          <div className="max-h-80 overflow-auto border border-border/70">
            <table className="min-w-[780px] w-full text-xs">
              <thead className="sticky top-0 bg-card text-left text-secondary-text">
                <tr>
                  <th className="px-3 py-2">{zh ? '信号日' : 'Signal'}</th>
                  <th className="px-3 py-2">{zh ? '成交日' : 'Execution'}</th>
                  <th className="px-3 py-2">{zh ? '标的' : 'Symbol'}</th>
                  <th className="px-3 py-2">{zh ? '操作' : 'Side'}</th>
                  <th className="px-3 py-2 text-right">{zh ? '数量' : 'Shares'}</th>
                  <th className="px-3 py-2 text-right">{zh ? '价格' : 'Price'}</th>
                  <th className="px-3 py-2 text-right">{zh ? '成本' : 'Cost'}</th>
                </tr>
              </thead>
              <tbody>
                {report.trades.map((trade, index) => (
                  <tr key={`${trade.executionDate}-${trade.symbol}-${index}`} className="border-t border-border/50">
                    <td className="px-3 py-2 text-secondary-text">{trade.signalDate}</td>
                    <td className="px-3 py-2 text-secondary-text">{trade.executionDate}</td>
                    <td className="px-3 py-2 font-mono text-foreground">{trade.symbol}</td>
                    <td className={`px-3 py-2 font-medium ${trade.action === 'buy' ? 'text-red-500' : 'text-emerald-600'}`}>{trade.action === 'buy' ? (zh ? '买入' : 'Buy') : (zh ? '卖出' : 'Sell')}</td>
                    <td className="px-3 py-2 text-right tabular-nums">{trade.shares}</td>
                    <td className="px-3 py-2 text-right tabular-nums">{number(trade.price, 3)}</td>
                    <td className="px-3 py-2 text-right tabular-nums">{number(trade.fee, 2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="border border-dashed border-border/70 px-4 py-8 text-center text-sm text-secondary-text">
            {zh ? '没有成交记录。' : 'No trades.'}
          </div>
        )}
      </section>
    </div>
  );
};

export default StrategyBacktestResults;
