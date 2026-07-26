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
  const labels = zh
    ? [
        ['交易策略总收益', pct(metrics.totalReturnPct), tone(metrics.totalReturnPct)],
        ['基准总收益', pct(metrics.benchmarkReturnPct), tone(metrics.benchmarkReturnPct)],
        ['Alpha（年化）', pct(metrics.alphaPct), tone(metrics.alphaPct)],
        ['Beta', number(metrics.beta), 'text-foreground'],
        ['Sharpe', number(metrics.sharpeRatio), tone(metrics.sharpeRatio)],
        ['最大回撤', pct(metrics.maxDrawdownPct), tone(metrics.maxDrawdownPct)],
        ['交易策略年化收益', pct(metrics.annualizedReturnPct), tone(metrics.annualizedReturnPct)],
        ['年化波动', pct(metrics.annualizedVolatilityPct), 'text-foreground'],
        ['日胜率', pct(metrics.winRatePct), tone(metrics.winRatePct - 50)],
        ['交易次数', String(metrics.tradeCount), 'text-foreground'],
      ]
    : [
        ['Strategy return', pct(metrics.totalReturnPct), tone(metrics.totalReturnPct)],
        ['Benchmark return', pct(metrics.benchmarkReturnPct), tone(metrics.benchmarkReturnPct)],
        ['Annualized alpha', pct(metrics.alphaPct), tone(metrics.alphaPct)],
        ['Beta', number(metrics.beta), 'text-foreground'],
        ['Sharpe', number(metrics.sharpeRatio), tone(metrics.sharpeRatio)],
        ['Maximum drawdown', pct(metrics.maxDrawdownPct), tone(metrics.maxDrawdownPct)],
        ['Annualized return', pct(metrics.annualizedReturnPct), tone(metrics.annualizedReturnPct)],
        ['Annualized volatility', pct(metrics.annualizedVolatilityPct), 'text-foreground'],
        ['Daily win rate', pct(metrics.winRatePct), tone(metrics.winRatePct - 50)],
        ['Trades', String(metrics.tradeCount), 'text-foreground'],
      ];

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
            {report.symbols.join(' / ')} · {report.period ? `${report.period.startDate} - ${report.period.endDate}` : '--'}
          </p>
        </div>
        <span className="rounded-md border border-border/70 bg-card/60 px-2 py-1 font-mono text-xs text-secondary-text">
          {report.status}
        </span>
      </div>

      <section className="grid grid-cols-2 gap-px overflow-hidden border border-border/70 bg-border/70 sm:grid-cols-3 xl:grid-cols-5">
        {labels.map(([label, value, className]) => (
          <div key={label} className="min-h-20 bg-card px-3 py-3">
            <p className="text-xs text-secondary-text">{label}</p>
            <p className={`mt-2 text-lg font-semibold tabular-nums ${className}`}>{value}</p>
          </div>
        ))}
      </section>

      {report.equityCurve.length ? (
        <section className="border border-border/70 bg-card/35 p-3">
          <h3 className="mb-3 text-sm font-medium text-foreground">
            {zh ? '交易策略与基准净值' : 'Trading strategy and benchmark equity'}
          </h3>
          <div className="h-72 min-w-0" data-testid="strategy-equity-chart">
            <ResponsiveContainer width="100%" height="100%" initialDimension={{ width: 640, height: 288 }}>
              <LineChart data={report.equityCurve} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.18)" />
                <XAxis dataKey="date" minTickGap={42} tick={{ fontSize: 10 }} />
                <YAxis tickFormatter={(value: number) => `${value.toFixed(0)}%`} tick={{ fontSize: 10 }} />
                <ChartTooltip formatter={value => `${Number(value).toFixed(2)}%`} />
                <Legend />
                <Line type="monotone" dataKey="strategyReturnPct" name={zh ? '交易策略' : 'Trading strategy'} stroke="#dc2626" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="benchmarkReturnPct" name={zh ? '基准' : 'Benchmark'} stroke="#2563eb" strokeWidth={1.8} dot={false} />
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
          <span className="text-xs text-secondary-text">{zh ? `累计成本 ${money(metrics.totalCost, language)}` : `Total cost ${money(metrics.totalCost, language)}`}</span>
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
