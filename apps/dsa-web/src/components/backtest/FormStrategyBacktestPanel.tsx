import type React from 'react';
import { useMemo, useState } from 'react';
import { Loader2, Play } from 'lucide-react';
import { backtestApi } from '../../api/backtest';
import { getParsedApiError } from '../../api/error';
import type { UiLanguage } from '../../i18n/uiText';
import type { FormStrategyId, StrategyBacktestResponse } from '../../types/backtest';
import StrategyBacktestResults from './StrategyBacktestResults';

const INPUT = 'input-surface input-focus-glow h-10 w-full rounded-lg border bg-transparent px-3 text-sm focus:outline-none disabled:opacity-60';

const STRATEGIES: Record<FormStrategyId, {
  zh: { name: string; rule: string; universe: string };
  en: { name: string; rule: string; universe: string };
}> = {
  momentum_rotation: {
    zh: {
      name: '创业板ETF-纳指ETF 20日动量轮动交易策略',
      rule: '默认使用 159915 创业板ETF和 513100 纳指ETF。每天收盘比较过去20个交易日涨幅，只选择涨幅第一且收盘价站上自身20日均线的ETF；持仓标的跌到第二或跌破20日均线时卖出，两者都不符合时空仓。信号在下一共同交易日开盘执行。',
      universe: '固定规则模板：159915、513100，涨幅回看20日，均线20日，持有第一名。修改下方标的或参数后，将按修改后的自定义规则回测。',
    },
    en: {
      name: 'ChiNext ETF-Nasdaq ETF 20-Day Momentum Rotation',
      rule: 'By default, compare the trailing 20-day returns of 159915 and 513100 after each close. Hold only the first-ranked ETF when it closes above its own 20-day moving average. Sell when the holding falls to second place or below its moving average; hold cash when neither qualifies. Execute at the next common open.',
      universe: 'Fixed-rule preset: 159915, 513100, 20-day return, 20-day moving average and Top 1. Editing the fields below creates a custom variant.',
    },
  },
  dual_moving_average: {
    zh: {
      name: '双均线择时',
      rule: '每天收盘后比较一只标的的短期均线和长期均线：短期均线在长期均线上方时全仓持有，否则空仓；下一交易日开盘执行。',
      universe: '只接受 1 个股票或 ETF。',
    },
    en: {
      name: 'Dual moving-average timing',
      rule: 'After each close, hold the single symbol when its fast moving average is above its slow moving average; otherwise hold cash. Execute at the next open.',
      universe: 'Exactly one stock or ETF.',
    },
  },
  buy_and_hold: {
    zh: {
      name: '等权买入持有',
      rule: '在回测区间第一个交易日开盘，将资金等权买入全部标的并一直持有，用作其他交易策略的对照。',
      universe: '1 个或多个股票、ETF。',
    },
    en: {
      name: 'Equal-weight buy and hold',
      rule: 'Buy all symbols at equal weights on the first open and hold them for the full period.',
      universe: 'One or more stocks or ETFs.',
    },
  },
};

function parseSymbols(value: string): string[] {
  return value.split(/[\s,，;；]+/).map(item => item.trim()).filter(Boolean);
}

interface Props {
  language: UiLanguage;
}

const FormStrategyBacktestPanel: React.FC<Props> = ({ language }) => {
  const zh = language === 'zh';
  const [strategyId, setStrategyId] = useState<FormStrategyId>('momentum_rotation');
  const [symbols, setSymbols] = useState('159915, 513100');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [initialCapital, setInitialCapital] = useState('100000');
  const [sideCostPct, setSideCostPct] = useState('0');
  const [lotSize, setLotSize] = useState('100');
  const [lookbackDays, setLookbackDays] = useState('20');
  const [movingAverageDays, setMovingAverageDays] = useState('20');
  const [fastMaDays, setFastMaDays] = useState('5');
  const [slowMaDays, setSlowMaDays] = useState('20');
  const [topN, setTopN] = useState('1');
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<StrategyBacktestResponse | null>(null);
  const strategy = STRATEGIES[strategyId][language];
  const parsedSymbols = useMemo(() => parseSymbols(symbols), [symbols]);

  const run = async () => {
    const capital = Number(initialCapital);
    const cost = Number(sideCostPct);
    const lot = Number(lotSize);
    const lookback = Number(lookbackDays);
    const average = Number(movingAverageDays);
    const fast = Number(fastMaDays);
    const slow = Number(slowMaDays);
    const selectedTopN = Number(topN);
    if (!parsedSymbols.length) {
      setError(zh ? '至少输入一个标的代码。' : 'Enter at least one symbol.');
      return;
    }
    if (![capital, cost, lot, lookback, average, fast, slow, selectedTopN].every(Number.isFinite)) {
      setError(zh ? '参数必须是有效数字。' : 'Parameters must be valid numbers.');
      return;
    }
    setRunning(true);
    setError(null);
    setReport(null);
    try {
      setReport(await backtestApi.runFormStrategy({
        strategyId,
        symbols: parsedSymbols,
        startDate: startDate || undefined,
        endDate: endDate || undefined,
        initialCapital: capital,
        sideCostPct: cost,
        lotSize: lot,
        lookbackDays: lookback,
        movingAverageDays: average,
        fastMaDays: fast,
        slowMaDays: slow,
        topN: selectedTopN,
      }));
    } catch (reason) {
      setError(getParsedApiError(reason).message);
    } finally {
      setRunning(false);
    }
  };

  return (
    <main className="min-h-0 flex-1 overflow-y-auto p-3 sm:p-4">
      <div className="mx-auto max-w-[1600px] space-y-4">
        <section className="border border-border/70 bg-card/35 p-4">
          <div className="grid gap-4 xl:grid-cols-[minmax(380px,1.15fr)_minmax(0,1.85fr)]">
            <label className="space-y-1">
              <span className="text-xs font-medium text-secondary-text">{zh ? '交易策略' : 'Trading strategy'}</span>
              <select value={strategyId} onChange={event => setStrategyId(event.target.value as FormStrategyId)} className={INPUT} disabled={running}>
                {(Object.keys(STRATEGIES) as FormStrategyId[]).map(id => (
                  <option key={id} value={id}>{STRATEGIES[id][language].name}</option>
                ))}
              </select>
            </label>
            <div className="border-l-2 border-cyan/60 pl-3">
              <h2 className="text-sm font-semibold text-foreground">{strategy.name}</h2>
              <p className="mt-1 text-sm leading-6 text-secondary-text">{strategy.rule}</p>
              <p className="mt-1 text-xs text-muted-text">{strategy.universe}</p>
            </div>
          </div>

          <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-7">
            <label className="space-y-1 sm:col-span-2">
              <span className="text-xs text-secondary-text">{zh ? '标的代码（逗号分隔）' : 'Symbols (comma-separated)'}</span>
              <input value={symbols} onChange={event => setSymbols(event.target.value.toUpperCase())} className={INPUT} disabled={running} />
            </label>
            <label className="space-y-1"><span className="text-xs text-secondary-text">{zh ? '开始日期' : 'Start date'}</span><input type="date" value={startDate} onChange={event => setStartDate(event.target.value)} className={INPUT} disabled={running} /></label>
            <label className="space-y-1"><span className="text-xs text-secondary-text">{zh ? '结束日期' : 'End date'}</span><input type="date" value={endDate} onChange={event => setEndDate(event.target.value)} className={INPUT} disabled={running} /></label>
            <label className="space-y-1"><span className="text-xs text-secondary-text">{zh ? '初始资金' : 'Initial capital'}</span><input aria-label={zh ? '初始资金' : 'Initial capital'} type="number" min="1" value={initialCapital} onChange={event => setInitialCapital(event.target.value)} className={INPUT} disabled={running} /></label>
            <label className="space-y-1"><span className="text-xs text-secondary-text">{zh ? '单边成本（%）' : 'Per-side cost (%)'}</span><input aria-label={zh ? '单边成本（%）' : 'Per-side cost (%)'} type="number" min="0" max="5" step="0.01" value={sideCostPct} onChange={event => setSideCostPct(event.target.value)} className={INPUT} disabled={running} /></label>
            <label className="space-y-1"><span className="text-xs text-secondary-text">{zh ? '最小交易数量' : 'Lot size'}</span><input aria-label={zh ? '最小交易数量' : 'Lot size'} type="number" min="1" step="1" value={lotSize} onChange={event => setLotSize(event.target.value)} className={INPUT} disabled={running} /></label>
          </div>

          {strategyId === 'momentum_rotation' ? (
            <div className="mt-3 grid gap-3 sm:grid-cols-3">
              <label className="space-y-1"><span className="text-xs text-secondary-text">{zh ? '涨幅回看（交易日）' : 'Momentum lookback'}</span><input aria-label={zh ? '涨幅回看（交易日）' : 'Momentum lookback'} type="number" min="1" value={lookbackDays} onChange={event => setLookbackDays(event.target.value)} className={INPUT} disabled={running} /></label>
              <label className="space-y-1"><span className="text-xs text-secondary-text">{zh ? '均线周期（交易日）' : 'Moving-average days'}</span><input aria-label={zh ? '均线周期（交易日）' : 'Moving-average days'} type="number" min="1" value={movingAverageDays} onChange={event => setMovingAverageDays(event.target.value)} className={INPUT} disabled={running} /></label>
              <label className="space-y-1"><span className="text-xs text-secondary-text">{zh ? '持有排名前 N 个' : 'Hold top N'}</span><input aria-label={zh ? '持有排名前 N 个' : 'Hold top N'} type="number" min="1" value={topN} onChange={event => setTopN(event.target.value)} className={INPUT} disabled={running} /></label>
            </div>
          ) : null}
          {strategyId === 'dual_moving_average' ? (
            <div className="mt-3 grid gap-3 sm:grid-cols-2">
              <label className="space-y-1"><span className="text-xs text-secondary-text">{zh ? '短期均线' : 'Fast MA'}</span><input aria-label={zh ? '短期均线' : 'Fast MA'} type="number" min="1" value={fastMaDays} onChange={event => setFastMaDays(event.target.value)} className={INPUT} disabled={running} /></label>
              <label className="space-y-1"><span className="text-xs text-secondary-text">{zh ? '长期均线' : 'Slow MA'}</span><input aria-label={zh ? '长期均线' : 'Slow MA'} type="number" min="2" value={slowMaDays} onChange={event => setSlowMaDays(event.target.value)} className={INPUT} disabled={running} /></label>
            </div>
          ) : null}

          <div className="mt-4 flex items-center justify-between gap-3 border-t border-border/60 pt-3">
            <p className="text-xs text-secondary-text">
              {zh ? `已输入 ${parsedSymbols.length} 个标的；0% 成本表示不扣手续费和滑点。` : `${parsedSymbols.length} symbols; 0% means fees and slippage are not deducted.`}
            </p>
            <button type="button" onClick={() => void run()} disabled={running} className="btn-primary h-10 min-w-[112px] gap-2 whitespace-nowrap px-4">
              {running ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
              {running ? (zh ? '正在回测' : 'Running') : (zh ? '运行回测' : 'Run backtest')}
            </button>
          </div>
        </section>

        {error ? <div role="alert" className="border border-red-500/30 bg-red-500/8 px-4 py-3 text-sm text-red-600">{error}</div> : null}
        <StrategyBacktestResults language={language} report={report} emptyTitle={zh ? '请选择交易策略并运行' : 'Choose a trading strategy and run'} />
      </div>
    </main>
  );
};

export default FormStrategyBacktestPanel;
