import type React from 'react';
import { useEffect } from 'react';
import FormStrategyBacktestPanel from '../components/backtest/FormStrategyBacktestPanel';
import { useUiLanguage } from '../contexts/UiLanguageContext';

const BacktestPage: React.FC = () => {
  const { language } = useUiLanguage();
  const zh = language === 'zh';

  useEffect(() => {
    document.title = zh ? '条件策略回测 - DSA' : 'Form Strategy Backtest - DSA';
  }, [zh]);

  return (
    <div className="flex min-h-full flex-col bg-transparent">
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-border/70 px-3 py-3 sm:px-4">
        <div>
          <h1 className="text-lg font-semibold text-foreground">
            {zh ? '条件策略回测' : 'Form Strategy Backtest'}
          </h1>
        </div>
      </header>
      <FormStrategyBacktestPanel language={language} />
    </div>
  );
};

export default BacktestPage;
