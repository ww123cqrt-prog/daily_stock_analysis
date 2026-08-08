import type React from 'react';
import { AlertTriangle, Database, ShieldCheck } from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type {
  EtfDefensiveRotationMetrics,
  EtfDefensiveRotationReport as EtfDefensiveRotationReportData,
} from '../../api/etfMomentum';
import { Badge, InlineAlert } from '../common';

const POSITIVE = '#dc2626';
const BLUE = '#2563eb';
const AMBER = '#d97706';
const TEAL = '#0f766e';
const GRAY = '#64748b';
const GRID = 'hsl(var(--border) / 0.45)';

const COPY = {
  zh: {
    previewWarning: '研究预览，不可用于交易',
    previewMessage: '当前 51 只 ETF 测试快照尚未完成 quant_data 正式验收。页面只展示固定规则研究结果，不生成候选评级，也不连接券商或下单。',
    overview: '规则与当前状态',
    rule: '固定规则',
    ruleText: '每天收盘先筛出站上自身 20 日均线的 ETF，再在合格标的中选择 20 日涨幅第一；下一共同交易日开盘执行。没有标的合格时持有现金。',
    noSearch: '未搜索参数',
    noSearchText: '回看期和均线均固定为 20 日，没有从邻近参数中挑选最好结果。',
    execution: '信号与执行',
    executionText: 'T 日收盘生成信号，T+1 三只 ETF 共同开盘执行。',
    latestSignal: '最新研究信号',
    signalDate: '信号日期',
    currentPosition: '最新开盘持仓',
    nextTarget: '下一共同开盘目标',
    unavailable: '不可用于交易',
    symbols: '参与轮动',
    cost: '费用口径',
    feeDefinition: '单边佣金 0.5bp = 成交金额的 0.005%',
    feeDetail: '买入和卖出分别计算；ETF 不计印花税和过户费。压力测试另加单边 1bp、2bp 滑点。',
    core: '核心结果',
    annualized: '年化收益',
    totalReturn: '总收益',
    maxDrawdown: '最大回撤',
    annualizedVolatility: '年化波动',
    sharpe: '夏普比率（无风险利率 0%）',
    calmar: 'Calmar 比率',
    positiveDays: '正收益日比例',
    bestDay: '最佳单日',
    worstDay: '最差单日',
    dataStatus: '数据状态',
    available: '可计算',
    unavailableData: '数据不可用',
    switches: '换仓次数',
    invested: '持仓比例',
    tradingDays: '个交易日',
    drawdownLimit: '研究上限',
    commonPeriod: '三只 ETF 共同区间',
    equity: '净值与比较基准',
    equityHelp: '所有曲线从 1 开始。表格列出全部比较对象，图中保留策略、旧两 ETF 规则、三 ETF 等权及大盘指数，便于阅读。',
    legacy: '旧两 ETF 固定规则',
    equalWeight: '三 ETF 等权',
    csi300: '沪深300',
    csi500: '中证500',
    csi800: '中证800',
    chinext: '创业板指',
    star50: '科创50',
    comparator: '比较对象',
    risk: '回撤',
    riskHelp: '绿色曲线表示策略净值距离此前最高点的跌幅。',
    stages: '分阶段表现',
    stage: '阶段',
    range: '区间',
    construction: '构建期',
    validation: '验证期',
    observation: '观察期',
    stability: '费用与稳定性',
    costStress: '成本压力测试',
    scenario: '场景',
    sideCost: '单边总成本',
    yearly: '年度收益',
    startSensitivity: '起始日期敏感性',
    regime: '市场状态归因',
    riskMetricNote: '夏普比率按日均净收益和年化波动计算，无风险利率固定为 0%；Calmar 比率=年化收益/最大回撤绝对值；正收益日只统计净收益大于 0 的交易日。',
    fullComparison: '策略、ETF 与大盘完整指标对比',
    periodComparison: '分阶段原始指标',
    costComparison: '成本压力原始指标',
    startComparison: '起始日期原始指标',
    leaveOneYear: '逐年留一原始指标',
    omittedYear: '剔除年份',
    remainingAnnualized: '剩余样本年化收益',
    legacyName: '原两 ETF 固定规则',
    eligibleName: '两 ETF 合格优先规则',
    holdings: '持仓行为',
    positionShare: '历史仓位比例',
    cash: '现金',
    data: '数据身份与研究边界',
    dataSource: 'ETF 数据快照',
    benchmarkSource: '指数数据快照',
    accepted: '正式验收',
    notAccepted: '未通过',
    missingBenchmark: '缺少的基准',
    noSubstitution: '没有使用 ETF 或其他指数替代缺失的创业板指。',
    generated: '生成时间',
    run: '研究运行',
    config: '配置哈希',
    git: '代码版本',
    historyWarning: '历史长度说明',
  },
  en: {
    previewWarning: 'Research preview, not for trading',
    previewMessage: 'The current 51-ETF test snapshot has not completed formal quant_data acceptance. This page reports a fixed-rule study only; it does not assign a candidate grade, connect to a broker, or place orders.',
    overview: 'Rules and current state',
    rule: 'Fixed rule',
    ruleText: 'At each close, first keep ETFs above their own 20-day moving average, then select the highest 20-day return among eligible assets. Execute at the next common open. Hold cash when none qualify.',
    noSearch: 'No parameter search',
    noSearchText: 'Both lookback and moving-average windows stay fixed at 20 days. No neighboring parameter was selected after seeing results.',
    execution: 'Signal and execution',
    executionText: 'Generate at the T close and execute at the next common open for all three ETFs.',
    latestSignal: 'Latest research signal',
    signalDate: 'Signal date',
    currentPosition: 'Position at latest open',
    nextTarget: 'Next common-open target',
    unavailable: 'Not for trading',
    symbols: 'Rotation universe',
    cost: 'Cost basis',
    feeDefinition: '0.5bp one-way commission = 0.005% of transaction value',
    feeDetail: 'Applied separately to buys and sells. No ETF stamp duty or transfer fee. Stress cases add 1bp or 2bp one-way slippage.',
    core: 'Core results',
    annualized: 'Annualized return',
    totalReturn: 'Total return',
    maxDrawdown: 'Maximum drawdown',
    annualizedVolatility: 'Annualized volatility',
    sharpe: 'Sharpe ratio (0% risk-free rate)',
    calmar: 'Calmar ratio',
    positiveDays: 'Positive-day ratio',
    bestDay: 'Best day',
    worstDay: 'Worst day',
    dataStatus: 'Data status',
    available: 'Available',
    unavailableData: 'Data unavailable',
    switches: 'Switches',
    invested: 'Invested share',
    tradingDays: 'trading days',
    drawdownLimit: 'Research limit',
    commonPeriod: 'Common three-ETF period',
    equity: 'Equity and benchmarks',
    equityHelp: 'All series start at 1. The table lists every comparator; the chart keeps the strategy, legacy two-ETF rule, equal weight, and broad indexes readable.',
    legacy: 'Legacy two-ETF rule',
    equalWeight: 'Three-ETF equal weight',
    csi300: 'CSI 300',
    csi500: 'CSI 500',
    csi800: 'CSI 800',
    chinext: 'ChiNext Index',
    star50: 'STAR 50',
    comparator: 'Comparator',
    risk: 'Drawdown',
    riskHelp: 'The green curve is the decline from the strategy\'s previous equity high.',
    stages: 'Performance by stage',
    stage: 'Stage',
    range: 'Range',
    construction: 'Construction',
    validation: 'Validation',
    observation: 'Observation',
    stability: 'Cost and stability',
    costStress: 'Cost stress',
    scenario: 'Scenario',
    sideCost: 'One-way total cost',
    yearly: 'Annual returns',
    startSensitivity: 'Start-date sensitivity',
    regime: 'Market-regime attribution',
    riskMetricNote: 'Sharpe uses daily net returns and annualized volatility with a 0% risk-free rate. Calmar is annualized return divided by absolute maximum drawdown. Positive-day ratio counts only net-return days above zero.',
    fullComparison: 'Complete raw-metric comparison',
    periodComparison: 'Raw metrics by stage',
    costComparison: 'Raw metrics under cost stress',
    startComparison: 'Raw metrics by start date',
    leaveOneYear: 'Raw leave-one-year metrics',
    omittedYear: 'Omitted year',
    remainingAnnualized: 'Annualized return of remaining sample',
    legacyName: 'Legacy two-ETF rule',
    eligibleName: 'Two-ETF eligible-first rule',
    holdings: 'Holding behavior',
    positionShare: 'Historical position share',
    cash: 'Cash',
    data: 'Data identity and research boundary',
    dataSource: 'ETF data snapshot',
    benchmarkSource: 'Index data snapshot',
    accepted: 'Formal acceptance',
    notAccepted: 'Not accepted',
    missingBenchmark: 'Missing benchmark',
    noSubstitution: 'No ETF or other index substitutes for the missing ChiNext index.',
    generated: 'Generated',
    run: 'Research run',
    config: 'Config hash',
    git: 'Code version',
    historyWarning: 'History-length note',
  },
} as const;

const COMPARATOR_LABELS: Record<string, string> = {
  SZ159915: '159915 创业板 ETF',
  SH513100: '513100 纳指 ETF',
  SH512890: '512890 红利低波 ETF',
  threeEtfEqualWeight: '三只 ETF 等权持有',
  twoEtfLegacyExact: '旧两 ETF 固定规则',
  twoEtfEligibleFirst: '两 ETF 合格优先规则',
  SH000300: '沪深300',
  SH000905: '中证500',
  SH000906: '中证800',
  SZ399006: '创业板指',
  SH000688: '科创50',
};

function pct(value: number | null | undefined, digits = 2): string {
  if (value == null || !Number.isFinite(value)) return '--';
  return `${(value * 100).toFixed(digits)}%`;
}

function number(value: number | null | undefined, digits = 2): string {
  if (value == null || !Number.isFinite(value)) return '--';
  return value.toFixed(digits);
}

function metricColor(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return 'text-secondary-text';
  return value >= 0 ? 'text-[#dc2626]' : 'text-[#16a34a]';
}

function chartPercent(value: number | string): string {
  return `${(Number(value) * 100).toFixed(0)}%`;
}

function symbolCode(symbol: string): string {
  return symbol.replace(/^(SH|SZ)/, '');
}

function DefensiveSection({
  title,
  description,
  children,
}: {
  title: string;
  description?: string;
  children: React.ReactNode;
}) {
  return (
    <section className="etf-report-section border-t border-border/60 py-7 sm:py-9">
      <div className="mb-5 max-w-4xl">
        <h2 className="text-xl font-semibold text-foreground sm:text-2xl">{title}</h2>
        {description ? <p className="mt-2 text-sm leading-6 text-secondary-text">{description}</p> : null}
      </div>
      {children}
    </section>
  );
}

function MetricBlock({ label, value, help }: { label: string; value: string; help: string }) {
  return (
    <div className="min-h-[118px] border-l-2 border-border/70 px-4 py-2 first:border-l-0 sm:first:border-l-2">
      <p className="text-xs font-medium text-muted-text">{label}</p>
      <p className="mt-2 font-mono text-2xl font-semibold text-foreground">{value}</p>
      <p className="mt-2 text-xs leading-5 text-secondary-text">{help}</p>
    </div>
  );
}

function comparatorLabel(key: string, strategyName: string): string {
  if (key === 'threeEtfDefensive') return strategyName;
  return COMPARATOR_LABELS[key] ?? key;
}

function metricRange(metrics: EtfDefensiveRotationMetrics): string {
  if (!metrics.startDate || !metrics.endDate) return '--';
  return `${metrics.startDate} - ${metrics.endDate}`;
}

function costScenarioLabel(key: string, language: 'zh' | 'en'): string {
  const match = key.match(/^(\d+)[Bb]pSlippage$/);
  if (!match) return key;
  return language === 'zh'
    ? `额外滑点 ${match[1]}bp`
    : `${match[1]}bp extra slippage`;
}

const COMPARISON_ORDER = [
  'threeEtfDefensive',
  'twoEtfLegacyExact',
  'twoEtfEligibleFirst',
  'SZ159915',
  'SH513100',
  'SH512890',
  'threeEtfEqualWeight',
  'SH000300',
  'SH000905',
  'SH000906',
  'SZ399006',
  'SH000688',
] as const;

type Copy = { [Key in keyof typeof COPY.zh]: string };

function RawMetricHeaders({ copy }: { copy: Copy }) {
  return (
    <>
      <th className="px-3 py-3 text-right font-medium">{copy.tradingDays}</th>
      <th className="px-3 py-3 text-right font-medium">{copy.annualized}</th>
      <th className="px-3 py-3 text-right font-medium">{copy.totalReturn}</th>
      <th className="px-3 py-3 text-right font-medium">{copy.maxDrawdown}</th>
      <th className="px-3 py-3 text-right font-medium">{copy.annualizedVolatility}</th>
      <th className="px-3 py-3 text-right font-medium">{copy.sharpe}</th>
      <th className="px-3 py-3 text-right font-medium">{copy.calmar}</th>
      <th className="px-3 py-3 text-right font-medium">{copy.positiveDays}</th>
      <th className="px-3 py-3 text-right font-medium">{copy.bestDay}</th>
      <th className="py-3 pl-3 text-right font-medium">{copy.worstDay}</th>
    </>
  );
}

function RawMetricCells({ metrics }: { metrics: EtfDefensiveRotationMetrics }) {
  return (
    <>
      <td className="px-3 py-3 text-right font-mono text-secondary-text">{metrics.periods ?? '--'}</td>
      <td className={`px-3 py-3 text-right font-mono ${metricColor(metrics.annualizedReturn)}`}>{pct(metrics.annualizedReturn)}</td>
      <td className={`px-3 py-3 text-right font-mono ${metricColor(metrics.totalReturn)}`}>{pct(metrics.totalReturn)}</td>
      <td className="px-3 py-3 text-right font-mono text-[#16a34a]">{pct(metrics.maxDrawdown)}</td>
      <td className="px-3 py-3 text-right font-mono text-foreground">{pct(metrics.annualizedVolatility)}</td>
      <td className={`px-3 py-3 text-right font-mono ${metricColor(metrics.sharpeRatio)}`}>{number(metrics.sharpeRatio)}</td>
      <td className={`px-3 py-3 text-right font-mono ${metricColor(metrics.calmarRatio)}`}>{number(metrics.calmarRatio)}</td>
      <td className="px-3 py-3 text-right font-mono text-foreground">{pct(metrics.positiveDayRatio)}</td>
      <td className={`px-3 py-3 text-right font-mono ${metricColor(metrics.bestDay)}`}>{pct(metrics.bestDay)}</td>
      <td className={`py-3 pl-3 text-right font-mono ${metricColor(metrics.worstDay)}`}>{pct(metrics.worstDay)}</td>
    </>
  );
}

function unavailableMetrics(): EtfDefensiveRotationMetrics {
  return {
    status: 'unavailable',
    annualizedReturn: null,
    totalReturn: null,
    maxDrawdown: null,
    annualizedVolatility: null,
    sharpeRatio: null,
    calmarRatio: null,
    positiveDayRatio: null,
    bestDay: null,
    worstDay: null,
  };
}

type Props = {
  report: EtfDefensiveRotationReportData;
  language: 'zh' | 'en';
};

const EtfDefensiveRotationReport: React.FC<Props> = ({ report, language }) => {
  const copy = COPY[language];
  const metrics = report.performance.primaryMetrics;
  const strategyName = report.strategy.name;
  const periodOrder = ['construction', 'validation', 'observation'];
  const periods = periodOrder
    .map((key) => ({ key, value: report.performance.periods[key] }))
    .filter((item): item is { key: string; value: NonNullable<typeof item.value> } => Boolean(item.value));
  const comparison = {
    ...report.performance.comparators,
    threeEtfDefensive: metrics,
  } as Record<string, EtfDefensiveRotationMetrics>;
  for (const symbol of report.benchmarks.missing) {
    comparison[symbol] ??= unavailableMetrics();
  }
  const comparisonRows = COMPARISON_ORDER.flatMap((key) => {
    const rowMetrics = comparison[key];
    return rowMetrics ? [{ key, metrics: rowMetrics }] : [];
  });
  const availableBenchmarkSymbols = new Set(report.benchmarks.available.map(({ symbol }) => symbol));
  const benchmarkIsAvailable = (symbol: string) => (
    availableBenchmarkSymbols.has(symbol) && comparison[symbol]?.status !== 'unavailable'
  );
  const costScenarios = report.stability.costStressComparison
    ? Object.entries(report.stability.costStressComparison).map(([key, value]) => ({
      key,
      sideCostBps: value.sideCostBps,
      strategy: value.threeEtfDefensive,
      legacy: value.twoEtfLegacyExact,
    }))
    : Object.entries(report.stability.costStress).map(([key, value]) => ({
      key,
      sideCostBps: value.sideCostBps,
      strategy: value,
      legacy: undefined,
    }));
  const regimeComparison = report.stability.marketRegimeComparison;
  const leaveOneYear = report.stability.leaveOneYearOut ?? [];
  const feePercent = report.strategy.commissionBpsEachSide / 100;

  return (
    <>
      <div className="mt-5 space-y-3">
        <InlineAlert
          variant="danger"
          title={copy.previewWarning}
          message={copy.previewMessage}
        />
      </div>

      <DefensiveSection title={copy.overview}>
        <div className="grid gap-7 xl:grid-cols-[1.35fr_0.65fr]">
          <div className="grid gap-5 md:grid-cols-3">
            {[
              [copy.rule, copy.ruleText],
              [copy.execution, copy.executionText],
              [copy.noSearch, copy.noSearchText],
            ].map(([label, text]) => (
              <div key={label} className="border-t-2 border-border/70 pt-3">
                <p className="text-sm font-semibold text-foreground">{label}</p>
                <p className="mt-2 text-sm leading-6 text-secondary-text">{text}</p>
              </div>
            ))}
            <div className="md:col-span-3">
              <p className="text-xs font-medium text-muted-text">{copy.symbols}</p>
              <div className="mt-2 flex flex-wrap gap-x-5 gap-y-2 text-sm text-foreground">
                {report.strategy.symbols.map((symbol) => (
                  <span key={symbol} className="font-mono">{symbolCode(symbol)}</span>
                ))}
              </div>
            </div>
          </div>
          <div className="border-l-2 border-border/70 pl-5">
            <div className="flex flex-wrap items-center gap-2">
              <h3 className="text-sm font-semibold text-foreground">{copy.latestSignal}</h3>
              <Badge variant="danger">{copy.unavailable}</Badge>
            </div>
            <dl className="mt-4 grid grid-cols-[auto_1fr] gap-x-4 gap-y-3 text-sm">
              <dt className="text-muted-text">{copy.signalDate}</dt>
              <dd className="font-mono text-foreground">{report.latestSignal.asOfDate}</dd>
              <dt className="text-muted-text">{copy.currentPosition}</dt>
              <dd className="font-mono text-foreground">{report.latestSignal.positionAtLatestOpen}</dd>
              <dt className="text-muted-text">{copy.nextTarget}</dt>
              <dd className="font-mono font-semibold text-foreground">{report.latestSignal.signalForNextCommonOpen}</dd>
            </dl>
          </div>
        </div>
        <div className="mt-7 border-t border-border/60 pt-5">
          <p className="text-xs font-medium text-muted-text">{copy.cost}</p>
          <p className="mt-2 text-sm font-semibold text-foreground">{copy.feeDefinition}</p>
          <p className="mt-1 text-xs leading-5 text-secondary-text">{copy.feeDetail}</p>
          <p className="mt-1 font-mono text-xs text-muted-text">{report.strategy.commissionBpsEachSide}bp = {feePercent.toFixed(3)}%</p>
        </div>
      </DefensiveSection>

      <DefensiveSection title={copy.core}>
        <div className="grid gap-y-5 sm:grid-cols-2 xl:grid-cols-5">
          <MetricBlock label={copy.annualized} value={pct(metrics.annualizedReturn)} help={`${copy.commonPeriod}: ${report.summary.dataStart} - ${report.summary.dataEnd}`} />
          <MetricBlock label={copy.totalReturn} value={pct(metrics.totalReturn)} help={`${metrics.periods ?? '--'} ${copy.tradingDays}`} />
          <MetricBlock label={copy.maxDrawdown} value={pct(metrics.maxDrawdown)} help={`${copy.drawdownLimit} ${pct(-Math.abs(report.summary.maxDrawdownLimit))}`} />
          <MetricBlock label={copy.switches} value={String(metrics.switchCount ?? report.holdingBehavior.switchCount)} help={copy.executionText} />
          <MetricBlock label={copy.invested} value={pct(metrics.investedFraction)} help={`${copy.cash}: ${pct(report.holdingBehavior.positionFraction.CASH)}`} />
        </div>
      </DefensiveSection>

      <DefensiveSection title={copy.equity} description={copy.equityHelp}>
        <div className="h-[380px] w-full" data-testid="defensive-equity-chart">
          <ResponsiveContainer width="100%" height="100%" minWidth={0} initialDimension={{ width: 1000, height: 380 }}>
            <LineChart data={report.chartData.equity} margin={{ top: 14, right: 14, bottom: 8, left: 0 }}>
              <CartesianGrid stroke={GRID} vertical={false} />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} minTickGap={54} />
              <YAxis tick={{ fontSize: 11 }} width={54} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="strategy" name={strategyName} stroke={POSITIVE} strokeWidth={2.5} dot={false} />
              <Line type="monotone" dataKey="twoEtfLegacyExact" name={copy.legacy} stroke={BLUE} strokeWidth={1.5} dot={false} />
              <Line type="monotone" dataKey="twoEtfEligibleFirst" name={copy.eligibleName} stroke="#7c3aed" strokeWidth={1.2} dot={false} />
              <Line type="monotone" dataKey="threeEtfEqualWeight" name={copy.equalWeight} stroke={GRAY} strokeWidth={1.4} dot={false} />
              {benchmarkIsAvailable('SH000300') ? <Line type="monotone" dataKey="SH000300" name={copy.csi300} stroke={AMBER} strokeWidth={1.2} dot={false} connectNulls /> : null}
              {benchmarkIsAvailable('SH000905') ? <Line type="monotone" dataKey="SH000905" name={copy.csi500} stroke={TEAL} strokeWidth={1.2} dot={false} connectNulls /> : null}
              {benchmarkIsAvailable('SH000906') ? <Line type="monotone" dataKey="SH000906" name={copy.csi800} stroke="#9333ea" strokeWidth={1.2} dot={false} connectNulls /> : null}
              {benchmarkIsAvailable('SZ399006') ? <Line type="monotone" dataKey="SZ399006" name={copy.chinext} stroke="#0891b2" strokeWidth={1.2} dot={false} connectNulls /> : null}
              {benchmarkIsAvailable('SH000688') ? <Line type="monotone" dataKey="SH000688" name={copy.star50} stroke="#be123c" strokeWidth={1.2} dot={false} connectNulls /> : null}
            </LineChart>
          </ResponsiveContainer>
        </div>
        <p className="mt-6 text-sm leading-6 text-secondary-text">{copy.riskMetricNote}</p>
        <div className="mt-4 overflow-x-auto border-t border-border/60">
          <table aria-label={copy.fullComparison} className="w-full min-w-[1580px] text-left text-sm">
            <thead className="text-xs text-muted-text">
              <tr>
                <th className="py-3 pr-4 font-medium">{copy.comparator}</th>
                <th className="px-3 py-3 text-right font-medium">{copy.dataStatus}</th>
                <th className="px-3 py-3 font-medium">{copy.range}</th>
                <RawMetricHeaders copy={copy} />
              </tr>
            </thead>
            <tbody className="divide-y divide-border/50">
              {comparisonRows.map((row) => (
                <tr key={row.key}>
                  <td className={`py-3 pr-4 text-foreground ${row.key === 'threeEtfDefensive' ? 'font-semibold' : ''}`}>{comparatorLabel(row.key, strategyName)}</td>
                  <td className="px-3 py-3 text-right text-xs text-secondary-text">{row.metrics.status === 'unavailable' ? copy.unavailableData : copy.available}</td>
                  <td className="px-3 py-3 font-mono text-xs text-secondary-text">{metricRange(row.metrics)}</td>
                  <RawMetricCells metrics={row.metrics} />
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </DefensiveSection>

      <DefensiveSection title={copy.risk} description={copy.riskHelp}>
        <div className="h-[300px]" data-testid="defensive-drawdown-chart">
          <ResponsiveContainer width="100%" height="100%" minWidth={0} initialDimension={{ width: 1000, height: 300 }}>
            <LineChart data={report.chartData.drawdown} margin={{ top: 12, right: 12, bottom: 6, left: 0 }}>
              <CartesianGrid stroke={GRID} vertical={false} />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} minTickGap={54} />
              <YAxis tickFormatter={chartPercent} tick={{ fontSize: 11 }} width={52} />
              <Tooltip formatter={(value) => pct(Number(value))} />
              <Legend />
              <Line type="monotone" dataKey="strategy" name={strategyName} stroke={POSITIVE} strokeWidth={2.2} dot={false} />
              <Line type="monotone" dataKey="twoEtfLegacyExact" name={copy.legacyName} stroke={BLUE} strokeWidth={1.5} dot={false} connectNulls />
              <Line type="monotone" dataKey="twoEtfEligibleFirst" name={copy.eligibleName} stroke="#7c3aed" strokeWidth={1.2} dot={false} connectNulls />
              <Line type="monotone" dataKey="threeEtfEqualWeight" name={copy.equalWeight} stroke={GRAY} strokeWidth={1.2} dot={false} connectNulls />
              {benchmarkIsAvailable('SH000300') ? <Line type="monotone" dataKey="SH000300" name={copy.csi300} stroke={AMBER} strokeWidth={1.2} dot={false} connectNulls /> : null}
              {benchmarkIsAvailable('SH000905') ? <Line type="monotone" dataKey="SH000905" name={copy.csi500} stroke={TEAL} strokeWidth={1.2} dot={false} connectNulls /> : null}
              {benchmarkIsAvailable('SH000906') ? <Line type="monotone" dataKey="SH000906" name={copy.csi800} stroke="#9333ea" strokeWidth={1.2} dot={false} connectNulls /> : null}
              {benchmarkIsAvailable('SZ399006') ? <Line type="monotone" dataKey="SZ399006" name={copy.chinext} stroke="#0891b2" strokeWidth={1.2} dot={false} connectNulls /> : null}
              {benchmarkIsAvailable('SH000688') ? <Line type="monotone" dataKey="SH000688" name={copy.star50} stroke="#be123c" strokeWidth={1.2} dot={false} connectNulls /> : null}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </DefensiveSection>

      <DefensiveSection title={copy.periodComparison}>
        <div className="overflow-x-auto border-t border-border/60">
          <table className="w-full min-w-[1500px] text-left text-sm">
            <thead className="text-xs text-muted-text">
              <tr>
                <th className="py-3 pr-4 font-medium">{copy.stage}</th>
                <th className="px-4 py-3 font-medium">{copy.range}</th>
                <th className="px-3 py-3 font-medium">{copy.comparator}</th>
                <RawMetricHeaders copy={copy} />
              </tr>
            </thead>
            <tbody className="divide-y divide-border/50">
              {periods.flatMap(({ key, value }) => [
                { key: `${key}-three`, label: copy[key as 'construction' | 'validation' | 'observation'], range: `${value.start} - ${value.end}`, name: strategyName, metrics: value.strategy },
                { key: `${key}-legacy`, label: copy[key as 'construction' | 'validation' | 'observation'], range: `${value.start} - ${value.end}`, name: copy.legacyName, metrics: value.legacyTwoEtf },
              ]).map((row) => (
                <tr key={row.key}>
                  <td className="py-3 pr-4 font-medium text-foreground">{row.label}</td>
                  <td className="px-4 py-3 font-mono text-xs text-secondary-text">{row.range}</td>
                  <td className="px-3 py-3 text-foreground">{row.name}</td>
                  <RawMetricCells metrics={row.metrics} />
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </DefensiveSection>

      <DefensiveSection title={copy.stability}>
        <h3 className="text-sm font-semibold text-foreground">{copy.costComparison}</h3>
        <div className="mt-3 overflow-x-auto border-t border-border/60">
          <table className="w-full min-w-[1520px] text-left text-sm">
            <thead className="text-xs text-muted-text"><tr>
              <th className="py-3 pr-3 font-medium">{copy.scenario}</th>
              <th className="px-3 py-3 font-medium">{copy.comparator}</th>
              <th className="px-3 py-3 text-right font-medium">{copy.sideCost}</th>
              <RawMetricHeaders copy={copy} />
            </tr></thead>
            <tbody className="divide-y divide-border/50">
              {costScenarios.flatMap((scenario) => [
                <tr key={`${scenario.key}-strategy`}>
                  <td className="py-3 pr-3 font-mono text-foreground">{costScenarioLabel(scenario.key, language)}</td>
                  <td className="px-3 py-3 text-foreground">{strategyName}</td>
                  <td className="px-3 py-3 text-right font-mono text-foreground">{scenario.sideCostBps ?? '--'}bp</td>
                  <RawMetricCells metrics={scenario.strategy} />
                </tr>,
                scenario.legacy ? <tr key={`${scenario.key}-legacy`}>
                  <td className="py-3 pr-3 font-mono text-secondary-text">{costScenarioLabel(scenario.key, language)}</td>
                  <td className="px-3 py-3 text-foreground">{copy.legacyName}</td>
                  <td className="px-3 py-3 text-right font-mono text-foreground">{scenario.sideCostBps ?? '--'}bp</td>
                  <RawMetricCells metrics={scenario.legacy} />
                </tr> : null,
              ])}
            </tbody>
          </table>
        </div>
        <div className="mt-8">
          <h3 className="text-sm font-semibold text-foreground">{copy.yearly}</h3>
          <div className="mt-3 h-[280px]">
            <ResponsiveContainer width="100%" height="100%" minWidth={0} initialDimension={{ width: 1000, height: 280 }}>
              <BarChart data={report.chartData.yearlyReturns} margin={{ top: 10, right: 8, bottom: 4, left: 0 }}>
                <CartesianGrid stroke={GRID} vertical={false} />
                <XAxis dataKey="year" tick={{ fontSize: 11 }} />
                <YAxis tickFormatter={chartPercent} tick={{ fontSize: 11 }} width={48} />
                <Tooltip formatter={(value) => pct(Number(value))} />
                <Legend />
                <Bar dataKey="strategy" name={strategyName} fill={POSITIVE} radius={[2, 2, 0, 0]} />
                <Bar dataKey="twoEtfLegacyExact" name={copy.legacyName} fill={BLUE} radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </DefensiveSection>

      <DefensiveSection title={copy.startComparison}>
        <div className="h-[270px]">
          <ResponsiveContainer width="100%" height="100%" minWidth={0} initialDimension={{ width: 1000, height: 270 }}>
            <BarChart data={report.stability.startDateSensitivity} margin={{ top: 10, right: 8, bottom: 4, left: 0 }}>
              <CartesianGrid stroke={GRID} vertical={false} />
              <XAxis dataKey="actualStart" tick={{ fontSize: 10 }} minTickGap={20} />
              <YAxis tickFormatter={chartPercent} tick={{ fontSize: 11 }} width={48} />
              <Tooltip formatter={(value) => pct(Number(value))} />
              <Legend />
              <Bar dataKey="strategy.annualizedReturn" name={strategyName} fill={POSITIVE} radius={[2, 2, 0, 0]} />
              <Bar dataKey="legacyTwoEtf.annualizedReturn" name={copy.legacyName} fill={BLUE} radius={[2, 2, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-5 overflow-x-auto border-t border-border/60">
          <table className="w-full min-w-[1400px] text-left text-sm">
            <thead className="text-xs text-muted-text"><tr><th className="py-3 pr-4 font-medium">{copy.startSensitivity}</th><th className="px-3 py-3 font-medium">{copy.comparator}</th><RawMetricHeaders copy={copy} /></tr></thead>
            <tbody className="divide-y divide-border/50">
              {report.stability.startDateSensitivity.flatMap((item) => [
                <tr key={`${item.actualStart}-strategy`}><td className="py-3 pr-4 font-mono text-secondary-text">{item.actualStart}</td><td className="px-3 py-3 text-foreground">{strategyName}</td><RawMetricCells metrics={item.strategy} /></tr>,
                <tr key={`${item.actualStart}-legacy`}><td className="py-3 pr-4 font-mono text-secondary-text">{item.actualStart}</td><td className="px-3 py-3 text-foreground">{copy.legacyName}</td><RawMetricCells metrics={item.legacyTwoEtf} /></tr>,
              ])}
            </tbody>
          </table>
        </div>
      </DefensiveSection>

      <DefensiveSection title={copy.regime}>
        {regimeComparison?.regimes ? (
          <div className="overflow-x-auto border-t border-border/60">
            <table className="w-full min-w-[1400px] text-left text-sm">
              <thead className="text-xs text-muted-text"><tr><th className="py-3 pr-4 font-medium">{copy.regime}</th><th className="px-3 py-3 font-medium">{copy.comparator}</th><RawMetricHeaders copy={copy} /></tr></thead>
              <tbody className="divide-y divide-border/50">
                {Object.entries(regimeComparison.regimes).flatMap(([regime, values]) => [
                  <tr key={`${regime}-strategy`}><td className="py-3 pr-4 font-mono text-foreground">{regime}</td><td className="px-3 py-3 text-foreground">{strategyName}</td><RawMetricCells metrics={values.threeEtfDefensive} /></tr>,
                  <tr key={`${regime}-legacy`}><td className="py-3 pr-4 font-mono text-secondary-text">{regime}</td><td className="px-3 py-3 text-foreground">{copy.legacyName}</td><RawMetricCells metrics={values.twoEtfLegacyExact} /></tr>,
                ])}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-sm leading-6 text-secondary-text">{regimeComparison?.reason ?? report.stability.marketRegimeAttribution.reason ?? copy.missingBenchmark}</p>
        )}
      </DefensiveSection>

      <DefensiveSection title={copy.leaveOneYear}>
        <div className="overflow-x-auto border-t border-border/60">
          <table className="w-full min-w-[620px] text-left text-sm">
            <thead className="text-xs text-muted-text"><tr><th className="py-3 pr-4 font-medium">{copy.omittedYear}</th><th className="px-3 py-3 font-medium">{copy.comparator}</th><th className="py-3 pl-3 text-right font-medium">{copy.remainingAnnualized}</th></tr></thead>
            <tbody className="divide-y divide-border/50">
              {leaveOneYear.flatMap((item) => [
                <tr key={`${item.omittedYear}-strategy`}><td className="py-3 pr-4 font-mono text-foreground">{item.omittedYear}</td><td className="px-3 py-3 text-foreground">{strategyName}</td><td className={`py-3 pl-3 text-right font-mono ${metricColor(item.strategyAnnualizedReturn)}`}>{pct(item.strategyAnnualizedReturn)}</td></tr>,
                <tr key={`${item.omittedYear}-legacy`}><td className="py-3 pr-4 font-mono text-secondary-text">{item.omittedYear}</td><td className="px-3 py-3 text-foreground">{copy.legacyName}</td><td className={`py-3 pl-3 text-right font-mono ${metricColor(item.baselineAnnualizedReturn)}`}>{pct(item.baselineAnnualizedReturn)}</td></tr>,
              ])}
            </tbody>
          </table>
        </div>
      </DefensiveSection>

      <DefensiveSection title={copy.holdings}>
        <div className="grid gap-px overflow-hidden border border-border/60 bg-border/60 sm:grid-cols-2 lg:grid-cols-4">
          {Object.entries(report.holdingBehavior.positionFraction).map(([symbol, fraction]) => (
            <div key={symbol} className="bg-base p-4">
              <p className="font-mono text-sm text-muted-text">{symbol === 'CASH' ? copy.cash : symbol}</p>
              <p className="mt-2 font-mono text-xl font-semibold text-foreground">{pct(fraction)}</p>
            </div>
          ))}
        </div>
      </DefensiveSection>

      <DefensiveSection title={copy.data}>
        <div className="grid gap-7 lg:grid-cols-2">
          <div className="border-t-2 border-border/70 pt-4">
            <h3 className="flex items-center gap-2 text-sm font-semibold text-foreground">
              <Database className="h-4 w-4" />
              {copy.dataSource}
            </h3>
            <dl className="mt-4 grid grid-cols-[auto_1fr] gap-x-4 gap-y-3 text-sm">
              <dt className="text-muted-text">dataset</dt><dd className="font-mono text-foreground">{report.dataQuality.datasetId}</dd>
              <dt className="text-muted-text">snapshot</dt><dd className="break-all font-mono text-xs text-foreground">{report.dataQuality.snapshotId}</dd>
              <dt className="text-muted-text">{copy.accepted}</dt><dd className="font-semibold text-[#16a34a]">{copy.notAccepted}</dd>
              <dt className="text-muted-text">SHA-256</dt><dd className="break-all font-mono text-xs text-foreground">{report.dataQuality.contentSha256 ?? '--'}</dd>
            </dl>
          </div>
          <div className="border-t-2 border-border/70 pt-4">
            <h3 className="flex items-center gap-2 text-sm font-semibold text-foreground">
              <ShieldCheck className="h-4 w-4" />
              {copy.benchmarkSource}
            </h3>
            <p className="mt-4 font-mono text-xs leading-5 text-foreground">{report.benchmarks.dataIdentity?.snapshotId ?? '--'}</p>
            {report.benchmarks.missing.length > 0 ? (
              <div className="mt-4 flex items-start gap-2 text-sm text-secondary-text">
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-[#16a34a]" />
                <p>{copy.missingBenchmark}: <span className="font-mono">{report.benchmarks.missing.join(', ')}</span>. {copy.noSubstitution}</p>
              </div>
            ) : null}
          </div>
        </div>
        <div className="mt-7 border-t border-border/60 pt-5 text-xs leading-5 text-muted-text">
          <p>{copy.historyWarning}: {report.summary.historyWarning}</p>
          <p className="mt-1">{copy.generated}: {new Date(report.generatedAt).toLocaleString(language === 'zh' ? 'zh-CN' : 'en-US')} / {copy.run}: {report.runId}</p>
          <p className="mt-1">{copy.config}: {report.reproducibility.configSha256?.slice(0, 16) ?? '--'} / {copy.git}: {report.reproducibility.gitCommit?.slice(0, 12) ?? '--'}</p>
        </div>
      </DefensiveSection>
    </>
  );
};

export default EtfDefensiveRotationReport;
