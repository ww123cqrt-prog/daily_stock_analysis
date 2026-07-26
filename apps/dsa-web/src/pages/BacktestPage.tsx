import type React from 'react';
import { useEffect, useState } from 'react';
import { BrainCircuit, Braces, FileChartColumn, SlidersHorizontal } from 'lucide-react';
import { Link, useSearchParams } from 'react-router-dom';
import FormStrategyBacktestPanel from '../components/backtest/FormStrategyBacktestPanel';
import ModelStrategyBacktestPanel from '../components/backtest/ModelStrategyBacktestPanel';
import ScriptStrategyBacktestPanel from '../components/backtest/ScriptStrategyBacktestPanel';
import { useUiLanguage } from '../contexts/UiLanguageContext';

type BacktestMode = 'script' | 'form' | 'trained';

function resolveBacktestMode(value: string | null): BacktestMode {
  return value === 'form' || value === 'trained' ? value : 'script';
}

const BacktestPage: React.FC = () => {
  const { language } = useUiLanguage();
  const zh = language === 'zh';
  const [searchParams, setSearchParams] = useSearchParams();
  const [mode, setMode] = useState<BacktestMode>(() => resolveBacktestMode(searchParams.get('tab')));

  useEffect(() => {
    document.title = zh ? '交易策略回测工作台 - DSA' : 'Trading Strategy Backtest Workbench - DSA';
  }, [zh]);

  useEffect(() => {
    setMode(resolveBacktestMode(searchParams.get('tab')));
  }, [searchParams]);

  const selectMode = (nextMode: BacktestMode) => {
    setMode(nextMode);
    const next = new URLSearchParams(searchParams);
    next.set('tab', nextMode);
    setSearchParams(next, { replace: true });
  };

  return (
    <div className="flex min-h-full flex-col bg-transparent">
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-border/70 px-3 py-3 sm:px-4">
        <div>
          <h1 className="text-lg font-semibold text-foreground">
            {zh ? '交易策略回测工作台' : 'Trading Strategy Backtest Workbench'}
          </h1>
          <p className="mt-1 text-xs text-secondary-text">
            {zh ? '脚本交易策略、条件交易策略，以及选股模型与交易策略组合都在这里回测。' : 'Backtest scripted trading strategies, form-based trading strategies, and stock-selection model plus trading-strategy combinations here.'}
          </p>
        </div>
        <div className="flex w-full flex-col gap-2 sm:w-auto sm:flex-row sm:items-center">
          <Link
            to="/backtest/etf-momentum"
            className="btn-secondary h-9 justify-center gap-2 px-3 text-sm"
          >
            <FileChartColumn className="h-4 w-4" />
            {zh ? 'ETF 动量完整报告' : 'ETF momentum report'}
          </Link>
          <div className="grid w-full grid-cols-3 border border-border/70 bg-card/45 p-1 sm:inline-flex sm:w-auto" role="tablist" aria-label={zh ? '回测方式' : 'Backtest mode'}>
          <button
            type="button"
            role="tab"
            aria-selected={mode === 'script'}
            onClick={() => selectMode('script')}
            className={`inline-flex h-10 items-center justify-center gap-2 whitespace-nowrap px-2 text-xs sm:h-9 sm:px-3 sm:text-sm ${mode === 'script' ? 'bg-cyan/12 text-cyan' : 'text-secondary-text hover:text-foreground'}`}
          >
            <Braces className="hidden h-4 w-4 sm:block" />
            {zh ? '脚本交易策略' : 'Scripted strategy'}
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={mode === 'form'}
            onClick={() => selectMode('form')}
            className={`inline-flex h-10 items-center justify-center gap-2 whitespace-nowrap px-2 text-xs sm:h-9 sm:px-3 sm:text-sm ${mode === 'form' ? 'bg-cyan/12 text-cyan' : 'text-secondary-text hover:text-foreground'}`}
          >
            <SlidersHorizontal className="hidden h-4 w-4 sm:block" />
            {zh ? '条件交易策略' : 'Form-based strategy'}
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={mode === 'trained'}
            aria-label={zh ? '选股模型与交易策略' : 'Stock-selection model and trading strategy'}
            onClick={() => selectMode('trained')}
            className={`inline-flex h-10 items-center justify-center gap-2 whitespace-nowrap px-2 text-xs sm:h-9 sm:px-3 sm:text-sm ${mode === 'trained' ? 'bg-cyan/12 text-cyan' : 'text-secondary-text hover:text-foreground'}`}
          >
            <BrainCircuit className="hidden h-4 w-4 sm:block" />
            {zh ? (
              <><span className="flex flex-col leading-tight sm:hidden"><span>选股模型</span><span>交易策略</span></span><span className="hidden sm:inline">选股模型与交易策略</span></>
            ) : (
              <><span className="sm:hidden">Model + strategy</span><span className="hidden sm:inline">Stock-selection model + trading strategy</span></>
            )}
          </button>
          </div>
        </div>
      </header>

      {mode === 'script' ? <ScriptStrategyBacktestPanel language={language} /> : null}
      {mode === 'form' ? <FormStrategyBacktestPanel language={language} /> : null}
      {mode === 'trained' ? <ModelStrategyBacktestPanel language={language} /> : null}
    </div>
  );
};

export default BacktestPage;
