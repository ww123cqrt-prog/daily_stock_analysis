import type React from 'react';
import { useEffect, useState } from 'react';
import CodeMirror from '@uiw/react-codemirror';
import { python } from '@codemirror/lang-python';
import { EditorView } from '@codemirror/view';
import { useTheme } from 'next-themes';
import { Activity, Archive, ArchiveRestore, ChevronDown, Code2, Loader2, Play, RotateCcw, Save, ShieldCheck } from 'lucide-react';
import { backtestApi } from '../../api/backtest';
import { getParsedApiError } from '../../api/error';
import type { UiLanguage } from '../../i18n/uiText';
import type {
  BacktestRunSummary,
  StabilityTestResponse,
  StrategyAssetSummary,
  StrategyBacktestResponse,
} from '../../types/backtest';
import { Button } from '../common';
import { PRESET_SCRIPTS, SAMPLE_SCRIPT } from './presetScripts';
import StabilityResultPanel from './StabilityResultPanel';
import StrategyBacktestResults from './StrategyBacktestResults';
import StrategyRunHistory from './StrategyRunHistory';

const INPUT = 'input-surface input-focus-glow h-10 w-full rounded-lg border bg-transparent px-3 text-sm focus:outline-none disabled:opacity-60';

const STABILITY_INPUT = 'input-surface h-8 w-20 rounded-md border bg-transparent px-2 text-xs focus:outline-none disabled:opacity-60';


const editorThemeDark = EditorView.theme({
  '&': {
    backgroundColor: 'transparent',
    fontSize: '13px',
    height: '100%',
    color: '#e2e8f0',
  },
  '.cm-scroller': {
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
    lineHeight: '24px',
  },
  '.cm-gutters': {
    backgroundColor: '#0c1117',
    borderRight: '1px solid rgba(255, 255, 255, 0.08)',
    color: '#475569',
  },
  '.cm-activeLine': { backgroundColor: 'rgba(34, 211, 238, 0.05)' },
  '.cm-content': { caretColor: '#22d3ee', padding: '12px 0' },
  '.cm-cursor': { borderLeftColor: '#22d3ee' },
  '.cm-comment': { color: '#6a9955' },
  '.cm-keyword': { color: '#c678dd' },
  '.cm-def': { color: '#61afef' },
  '.cm-string': { color: '#ce9178' },
  '.cm-number': { color: '#d19a66' },
  '.cm-atom, .cm-bool': { color: '#d19a66' },
  '.cm-operator': { color: '#56b6c2' },
  '.cm-builtin': { color: '#e5c07b' },
  '.cm-typeName, .cm-className': { color: '#e5c07b' },
  '.cm-property, .cm-propertyName': { color: '#e2e8f0' },
  '.cm-punctuation': { color: '#94a3b8' },
}, { dark: true });

const editorThemeLight = EditorView.theme({
  '&': {
    backgroundColor: 'transparent',
    fontSize: '13px',
    height: '100%',
    color: '#1f2937',
  },
  '.cm-scroller': {
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
    lineHeight: '24px',
  },
  '.cm-gutters': {
    backgroundColor: '#f3f4f6',
    borderRight: '1px solid rgba(0, 0, 0, 0.08)',
    color: '#9ca3af',
  },
  '.cm-activeLine': { backgroundColor: 'rgba(59, 130, 246, 0.06)' },
  '.cm-content': { caretColor: '#2563eb', padding: '12px 0' },
  '.cm-cursor': { borderLeftColor: '#2563eb' },
  '.cm-comment': { color: '#008000' },
  '.cm-keyword': { color: '#a626a4' },
  '.cm-def': { color: '#4078f2' },
  '.cm-string': { color: '#a31515' },
  '.cm-number': { color: '#986801' },
  '.cm-atom, .cm-bool': { color: '#986801' },
  '.cm-operator': { color: '#0184bc' },
  '.cm-builtin': { color: '#e45649' },
  '.cm-typeName, .cm-className': { color: '#c18401' },
  '.cm-property, .cm-propertyName': { color: '#1f2937' },
  '.cm-punctuation': { color: '#6b7280' },
}, { dark: false });

const INDEX_CHOICES: Array<{ code: string; labelZh: string; labelEn: string }> = [
  { code: 'SH000300', labelZh: '沪深300', labelEn: 'CSI 300' },
  { code: 'SH000905', labelZh: '中证500', labelEn: 'CSI 500' },
  { code: 'SH000906', labelZh: '中证800', labelEn: 'CSI 800' },
  { code: 'SZ399006', labelZh: '创业板指', labelEn: 'ChiNext' },
  { code: 'SH000688', labelZh: '科创50', labelEn: 'STAR 50' },
];

const LIBRARY_VIEWS = [
  { id: 'active', labelZh: '主策略', labelEn: 'Main strategies' },
  { id: 'archived', labelZh: '历史归档', labelEn: 'Archive' },
  { id: 'presets', labelZh: '脚本模板', labelEn: 'Templates' },
] as const;

type LibraryView = typeof LIBRARY_VIEWS[number]['id'];

function parseSymbols(value: string): string[] {
  return value.split(/[\s,，;；]+/).map(item => item.trim()).filter(Boolean);
}

interface Props {
  language: UiLanguage;
}

const ScriptStrategyBacktestPanel: React.FC<Props> = ({ language }) => {
  const zh = language === 'zh';
  const { resolvedTheme } = useTheme();
  const isDark = resolvedTheme !== 'light';
  const [script, setScript] = useState(SAMPLE_SCRIPT);
  const [pool, setPool] = useState<'etf' | 'csi800' | 'all_a' | 'custom'>('etf');
  const [symbols, setSymbols] = useState('159915, 513100');
  const [benchIndexes, setBenchIndexes] = useState<string[]>([]);
  const [benchSymbols, setBenchSymbols] = useState('');
  const [compareStrategyIds, setCompareStrategyIds] = useState<string[]>([]);
  const [comparePresetIds, setComparePresetIds] = useState<string[]>([]);
  const [stabilityMinWindow, setStabilityMinWindow] = useState('252');
  const [stabilityIterations, setStabilityIterations] = useState('20');
  const [stabilitySeed, setStabilitySeed] = useState('');
  const [stabilityRunning, setStabilityRunning] = useState(false);
  const [stabilityResult, setStabilityResult] = useState<StabilityTestResponse | null>(null);
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [initialCapital, setInitialCapital] = useState('100000');
  const [sideCostPct, setSideCostPct] = useState('0');
  const [lotSize, setLotSize] = useState('100');
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<StrategyBacktestResponse | null>(null);

  // Saved script assets (rule-based models)
  const [strategies, setStrategies] = useState<StrategyAssetSummary[]>([]);
  const [archivedStrategies, setArchivedStrategies] = useState<StrategyAssetSummary[]>([]);
  const [libraryView, setLibraryView] = useState<LibraryView>('active');
  const [loadingStrategies, setLoadingStrategies] = useState(true);
  const [loadingStrategy, setLoadingStrategy] = useState(false);
  const [changingArchive, setChangingArchive] = useState(false);
  const [selectedStrategyId, setSelectedStrategyId] = useState<string>('');
  const [activePresetId, setActivePresetId] = useState<string>('');
  const [strategyName, setStrategyName] = useState('');
  const [savingStrategy, setSavingStrategy] = useState(false);

  const visibleStrategies = libraryView === 'archived' ? archivedStrategies : strategies;
  const selectedStrategy = [...strategies, ...archivedStrategies].find(item => item.id === selectedStrategyId);
  const libraryBusy = loadingStrategies || loadingStrategy || savingStrategy || changingArchive;

  // Durable run history
  const [runs, setRuns] = useState<BacktestRunSummary[]>([]);
  const [refreshingRuns, setRefreshingRuns] = useState(true);
  const [loadingRunId, setLoadingRunId] = useState<string | null>(null);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);

  const loadStrategies = async () => {
    setLoadingStrategies(true);
    try {
      const [current, archived] = await Promise.all([
        backtestApi.listStrategyAssets(),
        backtestApi.listStrategyAssets({ archived: true }),
      ]);
      setStrategies(current);
      setArchivedStrategies(archived);
    } catch (reason) {
      setError(getParsedApiError(reason).message);
    } finally {
      setLoadingStrategies(false);
    }
  };

  const loadRuns = async () => {
    setRefreshingRuns(true);
    try {
      setRuns(await backtestApi.listBacktestRuns({ limit: 20 }));
    } catch (reason) {
      setError(getParsedApiError(reason).message);
    } finally {
      setRefreshingRuns(false);
    }
  };

  useEffect(() => {
    let active = true;
    void Promise.all([
      backtestApi.listStrategyAssets(),
      backtestApi.listStrategyAssets({ archived: true }),
    ]).then(([current, archived]) => {
      if (!active) return;
      setStrategies(current);
      setArchivedStrategies(archived);
    }).catch(reason => {
      if (active) setError(getParsedApiError(reason).message);
    }).finally(() => {
      if (active) setLoadingStrategies(false);
    });
    void backtestApi.listBacktestRuns({ limit: 20 }).then(history => {
      if (active) setRuns(history);
    }).catch(reason => {
      if (active) setError(getParsedApiError(reason).message);
    }).finally(() => {
      if (active) setRefreshingRuns(false);
    });
    return () => { active = false; };
  }, []);

  const changeLibraryView = (view: LibraryView) => {
    setLibraryView(view);
    setSelectedStrategyId('');
    setActivePresetId('');
    setStrategyName('');
    setError(null);
  };

  const selectStrategy = async (value: string) => {
    setError(null);
    if (!value) {
      setSelectedStrategyId('');
      setActivePresetId('');
      setStrategyName('');
      return;
    }
    if (value.startsWith('preset:')) {
      const preset = PRESET_SCRIPTS.find(item => `preset:${item.id}` === value);
      if (preset) {
        setScript(preset.script);
        setActivePresetId(preset.id);
        setSelectedStrategyId('');
        setStrategyName('');
      }
      return;
    }
    setLoadingStrategy(true);
    try {
      const detail = await backtestApi.getStrategyAsset(value);
      setScript(detail.script);
      setSelectedStrategyId(detail.id);
      setActivePresetId('');
      setStrategyName(detail.name);
    } catch (reason) {
      setError(getParsedApiError(reason).message);
      await loadStrategies();
    } finally {
      setLoadingStrategy(false);
    }
  };

  const saveStrategy = async () => {
    const name = strategyName.trim();
    if (!name) {
      setError(zh ? '保存前请先填写名称。' : 'Enter a name before saving.');
      return;
    }
    if (!script.trim()) {
      setError(zh ? '交易策略脚本不能为空。' : 'Trading strategy script cannot be empty.');
      return;
    }
    setSavingStrategy(true);
    setError(null);
    try {
      const saved = selectedStrategyId
        ? await backtestApi.updateStrategyAsset(selectedStrategyId, { name, script })
        : await backtestApi.saveStrategyAsset({ name, script });
      setSelectedStrategyId(saved.id);
      setActivePresetId('');
      setStrategyName(saved.name);
      setLibraryView(saved.archived ? 'archived' : 'active');
      await loadStrategies();
    } catch (reason) {
      setError(getParsedApiError(reason).message);
    } finally {
      setSavingStrategy(false);
    }
  };

  const changeArchiveState = async () => {
    if (!selectedStrategy) return;
    setChangingArchive(true);
    setError(null);
    try {
      await backtestApi.updateStrategyAsset(selectedStrategy.id, {
        archived: !selectedStrategy.archived,
      });
      setSelectedStrategyId('');
      setActivePresetId('');
      setStrategyName('');
      await loadStrategies();
    } catch (reason) {
      setError(getParsedApiError(reason).message);
    } finally {
      setChangingArchive(false);
    }
  };

  const selectRun = async (run: BacktestRunSummary) => {
    setLoadingRunId(run.id);
    setError(null);
    try {
      const detail = await backtestApi.getBacktestRun(run.id);
      setReport(detail.report);
      setSelectedRunId(run.id);
    } catch (reason) {
      setError(getParsedApiError(reason).message);
    } finally {
      setLoadingRunId(null);
    }
  };

  const deleteRun = async (run: BacktestRunSummary) => {
    setError(null);
    try {
      await backtestApi.deleteBacktestRun(run.id);
      if (selectedRunId === run.id) {
        setSelectedRunId(null);
        setReport(null);
      }
      await loadRuns();
    } catch (reason) {
      setError(getParsedApiError(reason).message);
    }
  };

  const toggleBenchIndex = (code: string) => {
    setBenchIndexes(prev => (
      prev.includes(code) ? prev.filter(item => item !== code) : [...prev, code]
    ));
  };

  const compareCount = compareStrategyIds.length + comparePresetIds.length;

  const toggleCompareStrategy = (strategyId: string) => {
    setCompareStrategyIds(prev => (
      prev.includes(strategyId)
        ? prev.filter(item => item !== strategyId)
        : compareCount >= 3 ? prev : [...prev, strategyId]
    ));
  };

  const toggleComparePreset = (presetId: string) => {
    setComparePresetIds(prev => (
      prev.includes(presetId)
        ? prev.filter(item => item !== presetId)
        : compareCount >= 3 ? prev : [...prev, presetId]
    ));
  };

  const runStability = async () => {
    const capital = Number(initialCapital);
    const cost = Number(sideCostPct);
    const lot = Number(lotSize);
    const minWindow = Number(stabilityMinWindow);
    const iterations = Number(stabilityIterations);
    const seedValue = stabilitySeed.trim();
    if (!script.trim()) {
      setError(zh ? '交易策略脚本不能为空。' : 'Trading strategy script cannot be empty.');
      return;
    }
    if (!startDate || !endDate || startDate > endDate) {
      setError(zh ? '请选择有效的开始和结束日期。' : 'Select a valid start and end date.');
      return;
    }
    if (![capital, cost, lot, minWindow, iterations].every(Number.isFinite)) {
      setError(zh ? '参数必须是有效数字。' : 'Parameters must be valid numbers.');
      return;
    }
    setStabilityRunning(true);
    setError(null);
    setStabilityResult(null);
    try {
      const result = await backtestApi.runStabilityTest({
        script,
        pool,
        symbols: pool === 'custom' ? parseSymbols(symbols) : undefined,
        startDate: startDate || undefined,
        endDate: endDate || undefined,
        initialCapital: capital,
        sideCostPct: cost,
        lotSize: lot,
        minWindowDays: Math.round(minWindow),
        iterations: Math.round(iterations),
        seed: seedValue ? Number(seedValue) : undefined,
      });
      setStabilityResult(result);
    } catch (reason) {
      setError(getParsedApiError(reason).message);
    } finally {
      setStabilityRunning(false);
    }
  };

  const run = async () => {
    const parsedSymbols = parseSymbols(symbols);
    const capital = Number(initialCapital);
    const cost = Number(sideCostPct);
    const lot = Number(lotSize);
    if (pool === 'custom' && !parsedSymbols.length) {
      setError(zh ? '自定义股票池至少需要一个标的代码。' : 'Custom pool requires at least one symbol.');
      return;
    }
    if (!script.trim()) {
      setError(zh ? '交易策略脚本不能为空。' : 'Trading strategy script cannot be empty.');
      return;
    }
    if (!startDate || !endDate || startDate > endDate) {
      setError(zh ? '请选择有效的开始和结束日期。' : 'Select a valid start and end date.');
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
      const result = await backtestApi.runScriptStrategy({
        script,
        pool,
        symbols: pool === 'custom' ? parsedSymbols : undefined,
        benchmarks: [...benchIndexes, ...parseSymbols(benchSymbols)],
        comparisonStrategyIds: compareStrategyIds,
        comparisonPresets: comparePresetIds
          .map(presetId => PRESET_SCRIPTS.find(item => item.id === presetId))
          .filter((item): item is NonNullable<typeof item> => item !== undefined)
          .map(item => ({ name: zh ? item.nameZh : item.nameEn, script: item.script })),
        startDate: startDate || undefined,
        endDate: endDate || undefined,
        initialCapital: capital,
        sideCostPct: cost,
        lotSize: lot,
        strategyAssetId: selectedStrategyId || undefined,
      });
      setReport(result);
      setSelectedRunId(result.runId ?? null);
      await loadRuns();
    } catch (reason) {
      setError(getParsedApiError(reason).message);
    } finally {
      setRunning(false);
    }
  };

  return (
    <main className="min-h-0 flex-1 overflow-y-auto p-3 sm:p-4">
      <div className="mx-auto max-w-[1780px] space-y-3">
        <section className="border border-border/70 bg-card/35 p-3">
          <div
            role="tablist"
            aria-label={zh ? '策略库' : 'Strategy library'}
            className="mb-3 flex w-fit max-w-full border-b border-border/70"
            onKeyDown={event => {
              if (running || libraryBusy) return;
              const index = LIBRARY_VIEWS.findIndex(view => view.id === libraryView);
              const offset = event.key === 'ArrowRight' ? 1 : event.key === 'ArrowLeft' ? -1 : 0;
              if (!offset && event.key !== 'Home' && event.key !== 'End') return;
              event.preventDefault();
              const next = event.key === 'Home' ? 0 : event.key === 'End' ? LIBRARY_VIEWS.length - 1
                : (index + offset + LIBRARY_VIEWS.length) % LIBRARY_VIEWS.length;
              changeLibraryView(LIBRARY_VIEWS[next].id);
              event.currentTarget.querySelector<HTMLButtonElement>(`[data-view="${LIBRARY_VIEWS[next].id}"]`)?.focus();
            }}
          >
            {LIBRARY_VIEWS.map(view => (
              <button
                key={view.id}
                id={`strategy-library-tab-${view.id}`}
                type="button"
                role="tab"
                data-view={view.id}
                aria-selected={libraryView === view.id}
                aria-controls="strategy-library-panel"
                tabIndex={libraryView === view.id ? 0 : -1}
                disabled={running || libraryBusy}
                onClick={() => changeLibraryView(view.id)}
                className={`h-9 border-b-2 px-3 text-xs whitespace-nowrap transition-colors disabled:opacity-50 ${
                  libraryView === view.id ? 'border-cyan-500 text-foreground' : 'border-transparent text-secondary-text hover:text-foreground'
                }`}
              >
                {zh ? view.labelZh : view.labelEn}
              </button>
            ))}
          </div>
          <div
            id="strategy-library-panel"
            role="tabpanel"
            aria-labelledby={`strategy-library-tab-${libraryView}`}
            className="flex flex-wrap items-end gap-2"
          >
            <label className="min-w-[200px] flex-1 space-y-1">
              <span className="text-xs text-secondary-text">{zh ? '已保存的脚本' : 'Saved scripts'}</span>
              <select
                aria-label={zh ? '已保存的脚本' : 'Saved scripts'}
                value={activePresetId ? `preset:${activePresetId}` : selectedStrategyId}
                onChange={event => void selectStrategy(event.target.value)}
                disabled={running || libraryBusy}
                className={INPUT}
              >
                <option value="">{libraryView === 'archived' && !visibleStrategies.length
                  ? (zh ? '暂无历史归档' : 'No archived strategies')
                  : (zh ? '未保存的脚本' : 'Unsaved script')}</option>
                {libraryView === 'presets' ? <optgroup label={zh ? '预置' : 'Presets'}>
                  {PRESET_SCRIPTS.map(item => (
                    <option key={item.id} value={`preset:${item.id}`}>{zh ? item.nameZh : item.nameEn}</option>
                  ))}
                </optgroup> : <optgroup label={libraryView === 'archived'
                  ? (zh ? '历史归档' : 'Archive') : (zh ? '主策略' : 'Main strategies')}>
                  {visibleStrategies.map(item => (
                    <option key={item.id} value={item.id}>{item.name}</option>
                  ))}
                </optgroup>}
              </select>
            </label>
            <label className="min-w-[180px] flex-1 space-y-1">
              <span className="text-xs text-secondary-text">{zh ? '名称' : 'Name'}</span>
              <input
                aria-label={zh ? '名称' : 'Name'}
                value={strategyName}
                onChange={event => setStrategyName(event.target.value)}
                disabled={running || libraryBusy}
                placeholder={zh ? '输入名称后保存' : 'Name to save as'}
                className={INPUT}
              />
            </label>
            <button
              type="button"
              onClick={() => void saveStrategy()}
              disabled={running || libraryBusy || !strategyName.trim()}
              className="btn-primary h-10 shrink-0 gap-2 px-4"
            >
              {savingStrategy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Save className="h-4 w-4" />}
              {selectedStrategyId ? (zh ? '保存修改' : 'Update') : (zh ? '保存脚本' : 'Save script')}
            </button>
            <Button
              type="button"
              onClick={() => void changeArchiveState()}
              disabled={running || libraryBusy || !selectedStrategy}
              variant="secondary"
              className="h-10 w-10 shrink-0 rounded-lg px-0"
              aria-label={selectedStrategy?.archived
                ? (zh ? '恢复到主列表' : 'Restore to main list') : (zh ? '归档策略' : 'Archive strategy')}
              title={selectedStrategy?.archived
                ? (zh ? '恢复到主列表' : 'Restore to main list') : (zh ? '归档策略' : 'Archive strategy')}
            >
              {changingArchive ? <Loader2 className="h-4 w-4 shrink-0 animate-spin" />
                : selectedStrategy?.archived ? <ArchiveRestore className="h-4 w-4 shrink-0" /> : <Archive className="h-4 w-4 shrink-0" />}
            </Button>
          </div>
        </section>

        <section className="grid gap-3 border border-border/70 bg-card/35 p-3 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-[minmax(150px,0.8fr)_minmax(150px,0.8fr)_repeat(5,minmax(110px,0.7fr))_auto]">
          <label className="space-y-1">
            <span className="text-xs text-secondary-text">{zh ? '股票池' : 'Universe pool'}</span>
            <select
              aria-label={zh ? '股票池' : 'Universe pool'}
              value={pool}
              onChange={event => setPool(event.target.value as 'etf' | 'csi800' | 'all_a' | 'custom')}
              disabled={running}
              className={INPUT}
            >
              <option value="etf">{zh ? 'ETF 池' : 'ETF pool'}</option>
              <option value="csi800">{zh ? 'CSI800（PIT 成员）' : 'CSI800 (PIT members)'}</option>
              <option value="all_a" disabled>{zh ? '全 A（暂未开放）' : 'All A-shares (unavailable)'}</option>
              <option value="custom">{zh ? '自选股票' : 'Custom'}</option>
            </select>
          </label>
          <label className="space-y-1">
            <span className="text-xs text-secondary-text">
              {zh ? '标的代码（自选时启用，逗号分隔）' : 'Symbols (enabled for custom, comma-separated)'}
            </span>
            <input
              value={symbols}
              onChange={event => setSymbols(event.target.value.toUpperCase())}
              className={INPUT}
              disabled={running || pool !== 'custom'}
              placeholder={pool === 'custom' ? '600519, 159915' : zh ? '自选池可用' : 'Available for custom pool'}
            />
          </label>
          <label className="space-y-1"><span className="text-xs text-secondary-text">{zh ? '开始日期' : 'Start date'}</span><input type="date" lang="en" value={startDate} onChange={event => setStartDate(event.target.value)} className={INPUT} disabled={running} /></label>
          <label className="space-y-1"><span className="text-xs text-secondary-text">{zh ? '结束日期' : 'End date'}</span><input type="date" lang="en" value={endDate} onChange={event => setEndDate(event.target.value)} className={INPUT} disabled={running} /></label>
          <label className="space-y-1"><span className="text-xs text-secondary-text">{zh ? '初始资金' : 'Initial capital'}</span><input aria-label={zh ? '初始资金' : 'Initial capital'} type="number" min="1" value={initialCapital} onChange={event => setInitialCapital(event.target.value)} className={INPUT} disabled={running} /></label>
          <label className="space-y-1"><span className="text-xs text-secondary-text">{zh ? '单边费率（%，佣金+滑点）' : 'Fee per side (%)'}</span><input aria-label={zh ? '单边费率（%）' : 'Fee per side (%)'} type="number" min="0" max="5" step="0.01" value={sideCostPct} onChange={event => setSideCostPct(event.target.value)} className={INPUT} disabled={running} /></label>
          <label className="space-y-1"><span className="text-xs text-secondary-text">{zh ? '最小交易数量' : 'Lot size'}</span><input aria-label={zh ? '最小交易数量' : 'Lot size'} type="number" min="1" step="1" value={lotSize} onChange={event => setLotSize(event.target.value)} className={INPUT} disabled={running} /></label>
          <button type="button" onClick={() => void run()} disabled={running} className="btn-primary h-10 min-w-[112px] self-end gap-2 whitespace-nowrap px-4">
            {running ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            {running ? (zh ? '正在回测' : 'Running') : (zh ? '运行回测' : 'Run')}
          </button>
        </section>

        <section className="border border-border/70 bg-card/35 p-3">
          <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
            <span className="text-xs text-secondary-text">{zh ? '对比基准' : 'Compare against'}</span>
            {INDEX_CHOICES.map(item => (
              <label key={item.code} className="flex cursor-pointer items-center gap-1.5 text-xs text-foreground">
                <input
                  type="checkbox"
                  checked={benchIndexes.includes(item.code)}
                  onChange={() => toggleBenchIndex(item.code)}
                  disabled={running}
                  className="h-3.5 w-3.5 accent-cyan-500"
                />
                {zh ? item.labelZh : item.labelEn}
              </label>
            ))}
            <label className="flex min-w-[220px] flex-1 items-center gap-1.5">
              <span className="shrink-0 text-xs text-secondary-text">{zh ? '自选对比' : 'Custom'}</span>
              <input
                value={benchSymbols}
                onChange={event => setBenchSymbols(event.target.value.toUpperCase())}
                disabled={running}
                placeholder={zh ? '如 600519（逗号分隔）' : 'e.g. 600519 (comma-separated)'}
                className={INPUT}
              />
            </label>
          </div>

          <details className="mt-3 border-t border-border/50 pt-2">
            <summary className="group flex cursor-pointer select-none items-center gap-1.5 text-xs text-secondary-text transition-colors hover:text-foreground">
              <ChevronDown className="h-3.5 w-3.5 transition-transform group-open:rotate-180" />
              <span>{zh ? '策略对比' : 'Strategy compare'}</span>
              <span className="text-[11px] text-muted-text">
                {zh ? `已选 ${compareCount} / 3` : `${compareCount} / 3 selected`}
              </span>
            </summary>
            <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1.5">
              {PRESET_SCRIPTS.map(item => (
                <label key={`preset:${item.id}`} className="flex cursor-pointer items-center gap-1.5 text-xs text-foreground">
                  <input
                    type="checkbox"
                    checked={comparePresetIds.includes(item.id)}
                    onChange={() => toggleComparePreset(item.id)}
                    disabled={running}
                    className="h-3.5 w-3.5 accent-cyan-500"
                  />
                  {zh ? item.nameZh : item.nameEn}
                </label>
              ))}
              {strategies.map(item => (
                <label key={item.id} className="flex cursor-pointer items-center gap-1.5 text-xs text-foreground">
                  <input
                    type="checkbox"
                    checked={compareStrategyIds.includes(item.id)}
                    onChange={() => toggleCompareStrategy(item.id)}
                    disabled={running}
                    className="h-3.5 w-3.5 accent-cyan-500"
                  />
                  {item.name}
                </label>
              ))}
              {archivedStrategies.map(item => (
                <label key={item.id} className="flex cursor-pointer items-center gap-1.5 text-xs text-secondary-text">
                  <input
                    type="checkbox"
                    checked={compareStrategyIds.includes(item.id)}
                    onChange={() => toggleCompareStrategy(item.id)}
                    disabled={running}
                    className="h-3.5 w-3.5 accent-cyan-500"
                  />
                  {item.name}{zh ? '（历史）' : ' (archived)'}
                </label>
              ))}
            </div>
          </details>
        </section>

        <section className="border border-border/70 bg-card/35 p-3">
          <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
            <span className="text-xs text-secondary-text">{zh ? '稳定性测试' : 'Stability test'}</span>
            <label className="flex items-center gap-1.5 text-xs text-foreground">
              <span>{zh ? '最小时间段' : 'Min window'}</span>
              <input type="number" min="20" value={stabilityMinWindow} onChange={event => setStabilityMinWindow(event.target.value)} disabled={running || stabilityRunning} className={STABILITY_INPUT} />
              <span className="text-secondary-text">{zh ? '交易日' : 'days'}</span>
            </label>
            <label className="flex items-center gap-1.5 text-xs text-foreground">
              <span>{zh ? '次数' : 'Runs'}</span>
              <input type="number" min="1" max="100" value={stabilityIterations} onChange={event => setStabilityIterations(event.target.value)} disabled={running || stabilityRunning} className={`${STABILITY_INPUT} w-16`} />
            </label>
            <label className="flex items-center gap-1.5 text-xs text-foreground">
              <span>{zh ? '种子（可空）' : 'Seed'}</span>
              <input value={stabilitySeed} onChange={event => setStabilitySeed(event.target.value)} disabled={running || stabilityRunning} placeholder="42" className={`${STABILITY_INPUT} w-20`} />
            </label>
            <button
              type="button"
              onClick={() => void runStability()}
              disabled={running || stabilityRunning}
              className="btn-secondary h-9 gap-1.5 px-3 text-xs"
            >
              {stabilityRunning ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Activity className="h-3.5 w-3.5" />}
              {stabilityRunning ? (zh ? '测试中' : 'Testing') : (zh ? '运行稳定性测试' : 'Run stability test')}
            </button>
          </div>
        </section>

        {error ? <div role="alert" className="border border-red-500/30 bg-red-500/8 px-4 py-3 text-sm text-red-600">{error}</div> : null}

        <div className="grid min-h-[680px] gap-3 xl:grid-cols-[minmax(520px,0.95fr)_minmax(0,1.25fr)]">
          <section className="flex min-h-[640px] flex-col overflow-hidden border border-slate-200 bg-white text-slate-800 dark:border-slate-700 dark:bg-[#10151c] dark:text-slate-200">
            <div className="flex h-11 items-center justify-between border-b border-black/10 bg-slate-100 px-3 dark:border-white/10 dark:bg-[#151c25]">
              <div className="flex items-center gap-2 text-sm font-medium"><Code2 className="h-4 w-4 text-cyan-400" />strategy.py</div>
              <div className="flex items-center gap-2">
                <span className="hidden items-center gap-1 text-[11px] text-slate-500 sm:flex dark:text-slate-400"><ShieldCheck className="h-3.5 w-3.5" />{zh ? '静态检查 + 隔离运行' : 'Static check + sandbox'}</span>
                <button type="button" onClick={() => setScript(SAMPLE_SCRIPT)} disabled={running} className="inline-flex h-8 w-8 items-center justify-center border border-black/10 text-slate-600 hover:bg-black/5 dark:border-white/10 dark:text-slate-300 dark:hover:bg-white/5" aria-label={zh ? '恢复示例' : 'Reset sample'}><RotateCcw className="h-4 w-4" /></button>
              </div>
            </div>
            <div className="relative min-h-0 flex-1 overflow-hidden text-[13px] leading-6">
              <CodeMirror
                value={script}
                onChange={value => setScript(value)}
                extensions={[python(), isDark ? editorThemeDark : editorThemeLight]}
                theme={isDark ? editorThemeDark : editorThemeLight}
                editable={!running}
                aria-label={zh ? 'Python交易策略脚本' : 'Python trading strategy script'}
                basicSetup={{
                  lineNumbers: true,
                  foldGutter: false,
                  autocompletion: false,
                  highlightActiveLine: true,
                  highlightActiveLineGutter: true,
                }}
                className="h-full"
                style={{ height: '100%' }}
              />
            </div>
          </section>

          <div className="flex min-w-0 flex-col gap-3">
            <section className="min-w-0 border border-border/70 bg-background/40 p-3">
              <StrategyBacktestResults language={language} report={report} emptyTitle={zh ? '编写交易策略后运行回测' : 'Write a trading strategy and run'} />
            </section>

            {stabilityResult ? (
              <StabilityResultPanel result={stabilityResult} language={language} />
            ) : null}

            <StrategyRunHistory
              runs={runs}
              loadingRunId={loadingRunId}
              selectedRunId={selectedRunId}
              refreshing={refreshingRuns}
              onSelect={run => void selectRun(run)}
              onRefresh={() => void loadRuns()}
              onDelete={run => void deleteRun(run)}
              language={language}
            />
          </div>
        </div>
      </div>

    </main>
  );
};

export default ScriptStrategyBacktestPanel;
