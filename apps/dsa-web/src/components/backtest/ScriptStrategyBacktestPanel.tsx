import type React from 'react';
import { useMemo, useRef, useState } from 'react';
import { Code2, Loader2, Play, RotateCcw, ShieldCheck } from 'lucide-react';
import { backtestApi } from '../../api/backtest';
import { getParsedApiError } from '../../api/error';
import type { UiLanguage } from '../../i18n/uiText';
import type { StrategyBacktestResponse } from '../../types/backtest';
import StrategyBacktestResults from './StrategyBacktestResults';

const INPUT = 'input-surface input-focus-glow h-10 w-full rounded-lg border bg-transparent px-3 text-sm focus:outline-none disabled:opacity-60';

const SAMPLE_SCRIPT = `def initialize(context):
    context.lookback = 20
    context.ma_days = 20


def on_bar(context, data):
    ranked = data.rank_by_return(context.lookback)
    if not ranked:
        context.target_cash()
        return

    best = ranked[0]
    average = data.sma(best, context.ma_days)
    if average is not None and data.close(best) > average:
        context.target_equal([best])
    else:
        context.target_cash()
`;

function parseSymbols(value: string): string[] {
  return value.split(/[\s,，;；]+/).map(item => item.trim()).filter(Boolean);
}

interface Props {
  language: UiLanguage;
}

const ScriptStrategyBacktestPanel: React.FC<Props> = ({ language }) => {
  const zh = language === 'zh';
  const editorRef = useRef<HTMLTextAreaElement>(null);
  const [script, setScript] = useState(SAMPLE_SCRIPT);
  const [symbols, setSymbols] = useState('159915, 513100');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [initialCapital, setInitialCapital] = useState('100000');
  const [sideCostPct, setSideCostPct] = useState('0');
  const [lotSize, setLotSize] = useState('100');
  const [scrollTop, setScrollTop] = useState(0);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<StrategyBacktestResponse | null>(null);
  const lineNumbers = useMemo(() => script.split('\n').map((_, index) => index + 1), [script]);

  const handleEditorKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key !== 'Tab') return;
    event.preventDefault();
    const target = event.currentTarget;
    const next = `${script.slice(0, target.selectionStart)}    ${script.slice(target.selectionEnd)}`;
    const cursor = target.selectionStart + 4;
    setScript(next);
    requestAnimationFrame(() => {
      editorRef.current?.setSelectionRange(cursor, cursor);
    });
  };

  const run = async () => {
    const parsedSymbols = parseSymbols(symbols);
    const capital = Number(initialCapital);
    const cost = Number(sideCostPct);
    const lot = Number(lotSize);
    if (!parsedSymbols.length) {
      setError(zh ? '至少输入一个脚本可访问的标的代码。' : 'Enter at least one script symbol.');
      return;
    }
    if (!script.trim()) {
      setError(zh ? '交易策略脚本不能为空。' : 'Trading strategy script cannot be empty.');
      return;
    }
    if (![capital, cost, lot].every(Number.isFinite)) {
      setError(zh ? '资金、成本和最小交易数量必须是有效数字。' : 'Capital, cost and lot size must be valid numbers.');
      return;
    }
    setRunning(true);
    setError(null);
    setReport(null);
    try {
      setReport(await backtestApi.runScriptStrategy({
        script,
        symbols: parsedSymbols,
        startDate: startDate || undefined,
        endDate: endDate || undefined,
        initialCapital: capital,
        sideCostPct: cost,
        lotSize: lot,
      }));
    } catch (reason) {
      setError(getParsedApiError(reason).message);
    } finally {
      setRunning(false);
    }
  };

  return (
    <main className="min-h-0 flex-1 overflow-y-auto p-3 sm:p-4">
      <div className="mx-auto max-w-[1780px] space-y-3">
        <section className="grid gap-3 border border-border/70 bg-card/35 p-3 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-[minmax(260px,1.3fr)_repeat(5,minmax(120px,0.7fr))_auto]">
          <label className="space-y-1">
            <span className="text-xs text-secondary-text">{zh ? '脚本可访问的标的' : 'Script universe'}</span>
            <input value={symbols} onChange={event => setSymbols(event.target.value.toUpperCase())} className={INPUT} disabled={running} />
          </label>
          <label className="space-y-1"><span className="text-xs text-secondary-text">{zh ? '开始日期' : 'Start date'}</span><input type="date" value={startDate} onChange={event => setStartDate(event.target.value)} className={INPUT} disabled={running} /></label>
          <label className="space-y-1"><span className="text-xs text-secondary-text">{zh ? '结束日期' : 'End date'}</span><input type="date" value={endDate} onChange={event => setEndDate(event.target.value)} className={INPUT} disabled={running} /></label>
          <label className="space-y-1"><span className="text-xs text-secondary-text">{zh ? '初始资金' : 'Initial capital'}</span><input aria-label={zh ? '初始资金' : 'Initial capital'} type="number" min="1" value={initialCapital} onChange={event => setInitialCapital(event.target.value)} className={INPUT} disabled={running} /></label>
          <label className="space-y-1"><span className="text-xs text-secondary-text">{zh ? '单边成本（%）' : 'Per-side cost (%)'}</span><input aria-label={zh ? '单边成本（%）' : 'Per-side cost (%)'} type="number" min="0" max="5" step="0.01" value={sideCostPct} onChange={event => setSideCostPct(event.target.value)} className={INPUT} disabled={running} /></label>
          <label className="space-y-1"><span className="text-xs text-secondary-text">{zh ? '最小交易数量' : 'Lot size'}</span><input aria-label={zh ? '最小交易数量' : 'Lot size'} type="number" min="1" step="1" value={lotSize} onChange={event => setLotSize(event.target.value)} className={INPUT} disabled={running} /></label>
          <button type="button" onClick={() => void run()} disabled={running} className="btn-primary h-10 min-w-[112px] self-end gap-2 whitespace-nowrap px-4">
            {running ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            {running ? (zh ? '正在回测' : 'Running') : (zh ? '运行回测' : 'Run')}
          </button>
        </section>

        {error ? <div role="alert" className="border border-red-500/30 bg-red-500/8 px-4 py-3 text-sm text-red-600">{error}</div> : null}

        <div className="grid min-h-[680px] gap-3 xl:grid-cols-[minmax(520px,0.95fr)_minmax(0,1.25fr)]">
          <section className="flex min-h-[640px] flex-col overflow-hidden border border-slate-700 bg-[#10151c] text-slate-200">
            <div className="flex h-11 items-center justify-between border-b border-white/10 bg-[#151c25] px-3">
              <div className="flex items-center gap-2 text-sm font-medium"><Code2 className="h-4 w-4 text-cyan-400" />strategy.py</div>
              <div className="flex items-center gap-2">
                <span className="flex items-center gap-1 text-[11px] text-slate-400"><ShieldCheck className="h-3.5 w-3.5" />{zh ? '静态检查 + 隔离运行' : 'Static check + sandbox'}</span>
                <button type="button" onClick={() => setScript(SAMPLE_SCRIPT)} disabled={running} className="inline-flex h-8 w-8 items-center justify-center border border-white/10 text-slate-300 hover:bg-white/5" title={zh ? '恢复示例' : 'Reset sample'} aria-label={zh ? '恢复示例' : 'Reset sample'}><RotateCcw className="h-4 w-4" /></button>
              </div>
            </div>
            <div className="relative min-h-0 flex-1 overflow-hidden font-mono text-[13px] leading-6">
              <div className="absolute inset-y-0 left-0 w-12 overflow-hidden border-r border-white/8 bg-[#0c1117] text-right text-slate-600" aria-hidden="true">
                <div style={{ transform: `translateY(-${scrollTop}px)` }} className="py-3 pr-3">
                  {lineNumbers.map(line => <div key={line}>{line}</div>)}
                </div>
              </div>
              <textarea
                ref={editorRef}
                aria-label={zh ? 'Python交易策略脚本' : 'Python trading strategy script'}
                value={script}
                onChange={event => setScript(event.target.value)}
                onKeyDown={handleEditorKeyDown}
                onScroll={event => setScrollTop(event.currentTarget.scrollTop)}
                spellCheck={false}
                wrap="off"
                disabled={running}
                className="absolute inset-0 resize-none overflow-auto whitespace-pre bg-transparent py-3 pl-16 pr-4 font-mono text-[13px] leading-6 text-slate-200 caret-cyan-400 outline-none selection:bg-cyan-500/20 disabled:opacity-70"
              />
            </div>
            <div className="border-t border-white/10 bg-[#0c1117] px-3 py-3 text-xs leading-5 text-slate-400">
              <p className="font-medium text-slate-300">DSA Python API</p>
              <p className="mt-1 font-mono">initialize(context) · on_bar(context, data)</p>
              <p className="font-mono">data.close / open / history / returns / sma / rank_by_return</p>
              <p className="font-mono">context.target_percent / target_equal / target_cash / log</p>
            </div>
          </section>

          <section className="min-w-0 border border-border/70 bg-background/40 p-3">
            <StrategyBacktestResults language={language} report={report} emptyTitle={zh ? '编写交易策略后运行回测' : 'Write a trading strategy and run'} />
          </section>
        </div>
      </div>
    </main>
  );
};

export default ScriptStrategyBacktestPanel;
