/**
 * ResearchPage — 量化研究工作区
 *
 * 研究的第一环：Python 策略 → 回测 → 结果。
 * 直接承载脚本策略界面（编写、沙箱执行、回测结果、策略资产、历史运行）。
 */

import type React from 'react';
import { useEffect } from 'react';
import ScriptStrategyBacktestPanel from '../components/backtest/ScriptStrategyBacktestPanel';
import { useUiLanguage } from '../contexts/UiLanguageContext';

const ResearchPage: React.FC = () => {
  const { language } = useUiLanguage();
  const zh = language === 'zh';

  useEffect(() => {
    document.title = zh ? '量化研究 - DSA' : 'Quant Research - DSA';
  }, [zh]);

  return (
    <div className="flex min-h-full flex-col bg-transparent">
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-border/70 px-3 py-3 sm:px-4">
        <h1 className="text-lg font-semibold text-foreground">
          {zh ? '量化研究' : 'Quant Research'}
        </h1>
      </header>
      <ScriptStrategyBacktestPanel language={language} />
    </div>
  );
};

export default ResearchPage;
