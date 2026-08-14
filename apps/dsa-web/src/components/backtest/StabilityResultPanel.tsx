import type React from 'react';
import { Activity } from 'lucide-react';
import type { StabilityTestResponse } from '../../types/backtest';
import type { UiLanguage } from '../../i18n/uiText';

interface Props {
  result: StabilityTestResponse;
  language: UiLanguage;
}

function pct(value: unknown): string {
  const n = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(n) ? `${n.toFixed(2)}%` : '--';
}

function tone(value: unknown): string {
  const n = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(n) || n === 0) return 'text-foreground';
  return n > 0 ? 'text-red-500' : 'text-emerald-600';
}

const StabilityResultPanel: React.FC<Props> = ({ result, language }) => {
  const zh = language === 'zh';
  const summary = result.summary as Record<string, unknown>;
  if (result.status !== 'completed' || !result.samples.length) {
    return (
      <section className="border border-border/70 bg-background/40 p-3">
        <div className="flex items-center gap-2 text-sm font-medium text-foreground">
          <Activity className="h-4 w-4 text-cyan" />
          {zh ? '稳定性测试' : 'Stability test'}
        </div>
        <p className="mt-2 text-xs text-secondary-text">
          {zh ? '该脚本在全区间没有产生可执行信号，无法做稳定性测试。' : 'No executable signals; the stability test cannot run.'}
        </p>
      </section>
    );
  }

  const cards: Array<[string, string, string]> = zh
    ? [
        ['正收益占比（胜率）', pct(summary.winRatePct), tone(Number(summary.winRatePct) - 50)],
        ['收益均值', pct(summary.returnMeanPct), tone(summary.returnMeanPct)],
        ['收益中位数', pct(summary.returnMedianPct), tone(summary.returnMedianPct)],
        ['收益标准差', pct(summary.returnStdPct), 'text-foreground'],
        ['最好子段', pct(summary.returnBestPct), tone(summary.returnBestPct)],
        ['最差子段', pct(summary.returnWorstPct), tone(summary.returnWorstPct)],
        ['回撤均值', pct(summary.drawdownMeanPct), tone(summary.drawdownMeanPct)],
        ['回撤最差', pct(summary.drawdownWorstPct), tone(summary.drawdownWorstPct)],
      ]
    : [
        ['Win rate', pct(summary.winRatePct), tone(Number(summary.winRatePct) - 50)],
        ['Mean return', pct(summary.returnMeanPct), tone(summary.returnMeanPct)],
        ['Median return', pct(summary.returnMedianPct), tone(summary.returnMedianPct)],
        ['Return std', pct(summary.returnStdPct), 'text-foreground'],
        ['Best slice', pct(summary.returnBestPct), tone(summary.returnBestPct)],
        ['Worst slice', pct(summary.returnWorstPct), tone(summary.returnWorstPct)],
        ['Mean drawdown', pct(summary.drawdownMeanPct), tone(summary.drawdownMeanPct)],
        ['Worst drawdown', pct(summary.drawdownWorstPct), tone(summary.drawdownWorstPct)],
      ];

  return (
    <section className="border border-border/70 bg-background/40 p-3">
      <div className="mb-3 flex items-center gap-2">
        <Activity className="h-4 w-4 text-cyan" />
        <h3 className="text-sm font-medium text-foreground">
          {zh ? '时间均匀性 / 稳定性' : 'Time-slice stability'}
        </h3>
        <span className="text-xs text-secondary-text">
          {zh ? `${result.samples.length} 个随机子时间段` : `${result.samples.length} random sub-periods`}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-border/70 bg-border/70 sm:grid-cols-4">
        {cards.map(([label, value, className]) => (
          <div key={label} className="bg-card px-3 py-2.5">
            <p className="text-[11px] text-secondary-text">{label}</p>
            <p className={`mt-1 text-sm font-medium tabular-nums ${className}`}>{value}</p>
          </div>
        ))}
      </div>

      <div className="mt-3 max-h-64 overflow-auto border border-border/70">
        <table className="w-full text-xs">
          <thead className="sticky top-0 bg-card text-left text-secondary-text">
            <tr>
              <th className="px-3 py-2">{zh ? '时间段' : 'Period'}</th>
              <th className="px-3 py-2 text-right">{zh ? '天数' : 'Days'}</th>
              <th className="px-3 py-2 text-right">{zh ? '收益' : 'Return'}</th>
              <th className="px-3 py-2 text-right">{zh ? '基准' : 'Benchmark'}</th>
              <th className="px-3 py-2 text-right">{zh ? '最大回撤' : 'MaxDD'}</th>
              <th className="px-3 py-2 text-right">{zh ? '交易' : 'Trades'}</th>
              <th className="px-3 py-2 text-right">{zh ? '持仓' : 'Exposure'}</th>
            </tr>
          </thead>
          <tbody>
            {result.samples.map(sample => (
              <tr key={`${sample.startDate}-${sample.endDate}`} className="border-t border-border/50">
                <td className="px-3 py-2 text-secondary-text">{sample.startDate} → {sample.endDate}</td>
                <td className="px-3 py-2 text-right tabular-nums">{sample.tradingDays}</td>
                <td className={`px-3 py-2 text-right font-mono tabular-nums ${tone(sample.totalReturnPct)}`}>{pct(sample.totalReturnPct)}</td>
                <td className={`px-3 py-2 text-right font-mono tabular-nums ${tone(sample.benchmarkReturnPct)}`}>{pct(sample.benchmarkReturnPct)}</td>
                <td className="px-3 py-2 text-right font-mono tabular-nums text-secondary-text">{pct(sample.maxDrawdownPct)}</td>
                <td className="px-3 py-2 text-right tabular-nums">{sample.tradeCount}</td>
                <td className="px-3 py-2 text-right tabular-nums">{sample.exposurePct.toFixed(0)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
};

export default StabilityResultPanel;
