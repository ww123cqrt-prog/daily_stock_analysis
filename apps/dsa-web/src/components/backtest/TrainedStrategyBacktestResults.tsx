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
import type { TrainedStrategyBacktestResponse } from '../../types/backtest';

type Props = {
  language: 'zh' | 'en';
  report: TrainedStrategyBacktestResponse;
};

function pct(value?: number | null, digits = 2): string {
  return value == null || !Number.isFinite(value) ? '--' : `${(value * 100).toFixed(digits)}%`;
}

function number(value?: number | null, digits = 2): string {
  return value == null || !Number.isFinite(value) ? '--' : value.toFixed(digits);
}

function tone(value?: number | null): string {
  if (value == null || value === 0) return 'text-foreground';
  return value > 0 ? 'text-red-500' : 'text-emerald-600';
}

const TrainedStrategyBacktestResults: React.FC<Props> = ({ language, report }) => {
  const zh = language === 'zh';
  const metrics = report.metrics;
  const chartData = report.equityCurve.map(point => ({
    ...point,
    strategyPct: point.strategyCumulativeReturn * 100,
    benchmarkPct: point.benchmarkCumulativeReturn * 100,
  }));
  const cards = zh
    ? [
        ['交易策略总收益', pct(metrics.totalReturn), tone(metrics.totalReturn)],
        ['沪深300总收益', pct(metrics.benchmarkTotalReturn), tone(metrics.benchmarkTotalReturn)],
        ['交易策略年化收益', pct(metrics.annualReturn), tone(metrics.annualReturn)],
        ['沪深300年化收益', pct(metrics.benchmarkAnnualReturn), tone(metrics.benchmarkAnnualReturn)],
        ['扣成本后年化超额', pct(metrics.annualExcessReturn), tone(metrics.annualExcessReturn)],
        ['信息比率', number(metrics.informationRatio), tone(metrics.informationRatio)],
        ['夏普比率', number(metrics.sharpe), tone(metrics.sharpe)],
        ['最大回撤', pct(metrics.maxDrawdown), tone(metrics.maxDrawdown)],
        ['平均换手', pct(metrics.avgTurnover), 'text-foreground'],
        ['平均仓位', pct(metrics.avgPosition), 'text-foreground'],
      ]
    : [
        ['Strategy return', pct(metrics.totalReturn), tone(metrics.totalReturn)],
        ['CSI 300 return', pct(metrics.benchmarkTotalReturn), tone(metrics.benchmarkTotalReturn)],
        ['Annual return', pct(metrics.annualReturn), tone(metrics.annualReturn)],
        ['CSI 300 annual return', pct(metrics.benchmarkAnnualReturn), tone(metrics.benchmarkAnnualReturn)],
        ['Net annual excess', pct(metrics.annualExcessReturn), tone(metrics.annualExcessReturn)],
        ['Information ratio', number(metrics.informationRatio), tone(metrics.informationRatio)],
        ['Sharpe ratio', number(metrics.sharpe), tone(metrics.sharpe)],
        ['Maximum drawdown', pct(metrics.maxDrawdown), tone(metrics.maxDrawdown)],
        ['Average turnover', pct(metrics.avgTurnover), 'text-foreground'],
        ['Average exposure', pct(metrics.avgPosition), 'text-foreground'],
      ];

  return (
    <section className="space-y-4 border border-border/70 bg-card/30 p-4">
      <div className="flex flex-wrap items-start justify-between gap-3 border-b border-border/70 pb-3">
        <div>
          <div className="flex items-center gap-2">
            <CheckCircle2 className="h-4 w-4 text-cyan" />
            <h2 className="text-base font-semibold text-foreground">
              {zh ? '本次重新计算结果' : 'Newly recalculated result'}
            </h2>
          </div>
          <p className="mt-1 text-xs text-secondary-text">
            {report.period.startDate} - {report.period.endDate} · {metrics.days} {zh ? '个交易日' : 'trading days'}
          </p>
          <p className="mt-1 text-xs text-secondary-text">
            {zh ? '回测引擎：' : 'Backtest engine: '}
            {report.replayEngine === 'qlib_portana'
              ? (zh ? 'Qlib 原始组合回测' : 'Original Qlib portfolio backtest')
              : (zh ? 'DSA 冻结策略回测' : 'Frozen DSA strategy backtest')}
          </p>
        </div>
        <span className="border border-border/70 bg-card/60 px-2 py-1 font-mono text-xs text-secondary-text">
          {report.status}
        </span>
      </div>

      {report.exactMetricValidation?.status === 'pass' ? (
        <div className="flex items-start gap-2 border border-emerald-500/25 bg-emerald-500/8 px-3 py-2 text-xs text-emerald-700">
          <CheckCircle2 className="mt-0.5 h-4 w-4 flex-none" />
          <span>
            {zh
              ? `重放的年化超额、信息比率和超额回撤与训练 Run 保存值一致（最大绝对误差 ${report.exactMetricValidation.maxAbsDifference.toExponential(2)}）。`
              : `Annual excess return, information ratio, and excess drawdown match the saved training Run (maximum absolute difference ${report.exactMetricValidation.maxAbsDifference.toExponential(2)}).`}
          </span>
        </div>
      ) : null}

      <div className="grid grid-cols-2 gap-px border border-border/70 bg-border/70 sm:grid-cols-3 xl:grid-cols-5">
        {cards.map(([label, value, className]) => (
          <div key={label} className="min-h-20 bg-card px-3 py-3">
            <p className="text-xs text-secondary-text">{label}</p>
            <p className={`mt-2 text-lg font-semibold tabular-nums ${className}`}>{value}</p>
          </div>
        ))}
      </div>

      <div className="border border-border/70 bg-card/35 p-3">
        <h3 className="mb-3 text-sm font-medium text-foreground">
          {zh ? '交易策略与沪深300累计收益' : 'Trading strategy and CSI 300 cumulative return'}
        </h3>
        <div className="h-72 min-w-0" data-testid="trained-strategy-equity-chart">
          <ResponsiveContainer width="100%" height="100%" initialDimension={{ width: 640, height: 288 }}>
            <LineChart data={chartData} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.18)" />
              <XAxis dataKey="date" minTickGap={42} tick={{ fontSize: 10 }} />
              <YAxis tickFormatter={(value: number) => `${value.toFixed(0)}%`} tick={{ fontSize: 10 }} />
              <ChartTooltip formatter={value => `${Number(value).toFixed(2)}%`} />
              <Legend />
              <Line type="monotone" dataKey="strategyPct" name={zh ? '交易策略' : 'Trading strategy'} stroke="#dc2626" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="benchmarkPct" name={zh ? '沪深300' : 'CSI 300'} stroke="#2563eb" strokeWidth={1.8} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {report.warnings.map(warning => (
        <div key={warning} className="flex items-start gap-2 border border-amber-500/25 bg-amber-500/8 px-3 py-2 text-xs text-amber-700">
          <AlertTriangle className="mt-0.5 h-4 w-4 flex-none" />
          <span>{warning}</span>
        </div>
      ))}
    </section>
  );
};

export default TrainedStrategyBacktestResults;
