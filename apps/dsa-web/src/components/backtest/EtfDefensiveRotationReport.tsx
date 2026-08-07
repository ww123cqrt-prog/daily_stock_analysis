import type React from 'react';
import { AlertTriangle, Database, ShieldCheck } from 'lucide-react';
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
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
const NEGATIVE = '#16a34a';
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
    switches: '换仓次数',
    invested: '持仓比例',
    tradingDays: '个交易日',
    drawdownLimit: '研究上限',
    commonPeriod: '三只 ETF 共同区间',
    equity: '净值与比较基准',
    equityHelp: '所有曲线从 1 开始。表格列出全部比较对象，图中保留策略、旧两 ETF 规则、三 ETF 等权及大盘指数，便于阅读。',
    strategy: '三 ETF 策略',
    legacy: '旧两 ETF 固定规则',
    equalWeight: '三 ETF 等权',
    csi300: '沪深300',
    csi500: '中证500',
    comparator: '比较对象',
    risk: '回撤',
    riskHelp: '绿色曲线表示策略净值距离此前最高点的跌幅。',
    stages: '分阶段表现',
    stage: '阶段',
    range: '区间',
    strategyAnnual: '三 ETF 年化',
    legacyAnnual: '旧两 ETF 年化',
    annualDelta: '年化差值',
    construction: '构建期',
    validation: '验证期',
    observation: '观察期',
    evidence: '相对旧两 ETF 的增量证据',
    fullDelta: '全区间年化差值',
    validationDelta: '验证期年化差值',
    observationDelta: '观察期年化差值',
    bootstrap: '区块 Bootstrap 95% 区间',
    positiveYears: '逐年留一为正比例',
    stability: '费用与稳定性',
    costStress: '成本压力测试',
    scenario: '场景',
    sideCost: '单边总成本',
    yearly: '年度收益',
    startSensitivity: '起始日期敏感性',
    regime: '市场状态归因',
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
    switches: 'Switches',
    invested: 'Invested share',
    tradingDays: 'trading days',
    drawdownLimit: 'Research limit',
    commonPeriod: 'Common three-ETF period',
    equity: 'Equity and benchmarks',
    equityHelp: 'All series start at 1. The table lists every comparator; the chart keeps the strategy, legacy two-ETF rule, equal weight, and broad indexes readable.',
    strategy: 'Three-ETF strategy',
    legacy: 'Legacy two-ETF rule',
    equalWeight: 'Three-ETF equal weight',
    csi300: 'CSI 300',
    csi500: 'CSI 500',
    comparator: 'Comparator',
    risk: 'Drawdown',
    riskHelp: 'The green curve is the decline from the strategy\'s previous equity high.',
    stages: 'Performance by stage',
    stage: 'Stage',
    range: 'Range',
    strategyAnnual: 'Three-ETF annual',
    legacyAnnual: 'Legacy annual',
    annualDelta: 'Annual delta',
    construction: 'Construction',
    validation: 'Validation',
    observation: 'Observation',
    evidence: 'Incremental evidence versus legacy two-ETF rule',
    fullDelta: 'Full-period annual delta',
    validationDelta: 'Validation annual delta',
    observationDelta: 'Observation annual delta',
    bootstrap: 'Block-bootstrap 95% interval',
    positiveYears: 'Positive leave-one-year ratio',
    stability: 'Cost and stability',
    costStress: 'Cost stress',
    scenario: 'Scenario',
    sideCost: 'One-way total cost',
    yearly: 'Annual returns',
    startSensitivity: 'Start-date sensitivity',
    regime: 'Market-regime attribution',
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
};

function pct(value: number | null | undefined, digits = 2): string {
  if (value == null || !Number.isFinite(value)) return '--';
  return `${(value * 100).toFixed(digits)}%`;
}

function signedPct(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return '--';
  const sign = value > 0 ? '+' : '';
  return `${sign}${(value * 100).toFixed(2)}%`;
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

function comparatorLabel(key: string): string {
  return COMPARATOR_LABELS[key] ?? key;
}

function costScenarioLabel(key: string, language: 'zh' | 'en'): string {
  const match = key.match(/^(\d+)[Bb]pSlippage$/);
  if (!match) return key;
  return language === 'zh'
    ? `额外滑点 ${match[1]}bp`
    : `${match[1]}bp extra slippage`;
}

function metricsRows(metrics: Record<string, EtfDefensiveRotationMetrics>) {
  return Object.entries(metrics).map(([key, value]) => ({ key, ...value }));
}

type Props = {
  report: EtfDefensiveRotationReportData;
  language: 'zh' | 'en';
};

const EtfDefensiveRotationReport: React.FC<Props> = ({ report, language }) => {
  const copy = COPY[language];
  const metrics = report.performance.primaryMetrics;
  const evidence = report.performance.incrementalEvidenceVsLegacyTwoEtf;
  const periodOrder = ['construction', 'validation', 'observation'];
  const periods = periodOrder
    .map((key) => ({ key, value: report.performance.periods[key] }))
    .filter((item): item is { key: string; value: NonNullable<typeof item.value> } => Boolean(item.value));
  const costRows = metricsRows(report.stability.costStress);
  const comparatorRows = metricsRows(report.performance.comparators);
  const regimeAttribution = report.stability.marketRegimeAttribution;
  const regimeRows = metricsRows(regimeAttribution.regimes ?? {});
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
              <Line type="monotone" dataKey="strategy" name={copy.strategy} stroke={POSITIVE} strokeWidth={2.5} dot={false} />
              <Line type="monotone" dataKey="twoEtfLegacyExact" name={copy.legacy} stroke={BLUE} strokeWidth={1.5} dot={false} />
              <Line type="monotone" dataKey="threeEtfEqualWeight" name={copy.equalWeight} stroke={GRAY} strokeWidth={1.4} dot={false} />
              <Line type="monotone" dataKey="SH000300" name={copy.csi300} stroke={AMBER} strokeWidth={1.2} dot={false} connectNulls />
              <Line type="monotone" dataKey="SH000905" name={copy.csi500} stroke={TEAL} strokeWidth={1.2} dot={false} connectNulls />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-6 overflow-x-auto border-t border-border/60">
          <table className="w-full min-w-[680px] text-left text-sm">
            <thead className="text-xs text-muted-text">
              <tr>
                <th className="py-3 pr-4 font-medium">{copy.comparator}</th>
                <th className="px-4 py-3 text-right font-medium">{copy.annualized}</th>
                <th className="px-4 py-3 text-right font-medium">{copy.totalReturn}</th>
                <th className="py-3 pl-4 text-right font-medium">{copy.maxDrawdown}</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/50">
              <tr>
                <td className="py-3 pr-4 font-semibold text-foreground">{copy.strategy}</td>
                <td className="px-4 py-3 text-right font-mono text-[#dc2626]">{pct(metrics.annualizedReturn)}</td>
                <td className="px-4 py-3 text-right font-mono text-[#dc2626]">{pct(metrics.totalReturn)}</td>
                <td className="py-3 pl-4 text-right font-mono text-[#16a34a]">{pct(metrics.maxDrawdown)}</td>
              </tr>
              {comparatorRows.map((row) => (
                <tr key={row.key}>
                  <td className="py-3 pr-4 text-foreground">{comparatorLabel(row.key)}</td>
                  <td className={`px-4 py-3 text-right font-mono ${metricColor(row.annualizedReturn)}`}>{pct(row.annualizedReturn)}</td>
                  <td className={`px-4 py-3 text-right font-mono ${metricColor(row.totalReturn)}`}>{pct(row.totalReturn)}</td>
                  <td className="py-3 pl-4 text-right font-mono text-[#16a34a]">{pct(row.maxDrawdown)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </DefensiveSection>

      <DefensiveSection title={copy.risk} description={copy.riskHelp}>
        <div className="h-[300px]" data-testid="defensive-drawdown-chart">
          <ResponsiveContainer width="100%" height="100%" minWidth={0} initialDimension={{ width: 1000, height: 300 }}>
            <AreaChart data={report.chartData.drawdown} margin={{ top: 12, right: 12, bottom: 6, left: 0 }}>
              <CartesianGrid stroke={GRID} vertical={false} />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} minTickGap={54} />
              <YAxis tickFormatter={chartPercent} tick={{ fontSize: 11 }} width={52} />
              <Tooltip formatter={(value) => pct(Number(value))} />
              <Area type="monotone" dataKey="strategy" stroke={NEGATIVE} fill={NEGATIVE} fillOpacity={0.16} dot={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </DefensiveSection>

      <DefensiveSection title={copy.stages}>
        <div className="overflow-x-auto border-t border-border/60">
          <table className="w-full min-w-[720px] text-left text-sm">
            <thead className="text-xs text-muted-text">
              <tr>
                <th className="py-3 pr-4 font-medium">{copy.stage}</th>
                <th className="px-4 py-3 font-medium">{copy.range}</th>
                <th className="px-4 py-3 text-right font-medium">{copy.strategyAnnual}</th>
                <th className="px-4 py-3 text-right font-medium">{copy.legacyAnnual}</th>
                <th className="py-3 pl-4 text-right font-medium">{copy.annualDelta}</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/50">
              {periods.map(({ key, value }) => {
                const delta = (value.strategy.annualizedReturn ?? 0) - (value.legacyTwoEtf.annualizedReturn ?? 0);
                const stageLabel = copy[key as 'construction' | 'validation' | 'observation'];
                return (
                  <tr key={key}>
                    <td className="py-3 pr-4 font-medium text-foreground">{stageLabel}</td>
                    <td className="px-4 py-3 font-mono text-xs text-secondary-text">{value.start} - {value.end}</td>
                    <td className="px-4 py-3 text-right font-mono text-foreground">{pct(value.strategy.annualizedReturn)}</td>
                    <td className="px-4 py-3 text-right font-mono text-foreground">{pct(value.legacyTwoEtf.annualizedReturn)}</td>
                    <td className={`py-3 pl-4 text-right font-mono ${metricColor(delta)}`}>{signedPct(delta)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div className="mt-8">
          <h3 className="text-sm font-semibold text-foreground">{copy.evidence}</h3>
          <div className="mt-3 grid gap-px overflow-hidden border border-border/60 bg-border/60 sm:grid-cols-2 xl:grid-cols-5">
            {[
              [copy.fullDelta, signedPct(evidence.fullAnnualizedDelta), evidence.fullAnnualizedDelta],
              [copy.validationDelta, signedPct(evidence.validationAnnualizedDelta), evidence.validationAnnualizedDelta],
              [copy.observationDelta, signedPct(evidence.observationAnnualizedDelta), evidence.observationAnnualizedDelta],
              [copy.bootstrap, `${pct(report.stability.blockBootstrap.ci95Lower)} - ${pct(report.stability.blockBootstrap.ci95Upper)}`, report.stability.blockBootstrap.ci95Lower],
              [copy.positiveYears, pct(report.stability.leaveOneYearPositiveRatio), report.stability.leaveOneYearPositiveRatio],
            ].map(([label, value, tone]) => (
              <div key={String(label)} className="min-h-[92px] bg-base p-4">
                <p className="text-xs leading-5 text-muted-text">{label}</p>
                <p className={`mt-2 font-mono text-lg font-semibold ${metricColor(Number(tone))}`}>{value}</p>
              </div>
            ))}
          </div>
        </div>
      </DefensiveSection>

      <DefensiveSection title={copy.stability}>
        <div className="grid gap-8 xl:grid-cols-2">
          <div className="min-w-0">
            <h3 className="text-sm font-semibold text-foreground">{copy.costStress}</h3>
            <div className="mt-3 overflow-x-auto border-t border-border/60">
              <table className="w-full min-w-[480px] text-left text-sm">
                <thead className="text-xs text-muted-text">
                  <tr>
                    <th className="py-3 pr-3 font-medium">{copy.scenario}</th>
                    <th className="px-3 py-3 text-right font-medium">{copy.sideCost}</th>
                    <th className="px-3 py-3 text-right font-medium">{copy.annualized}</th>
                    <th className="py-3 pl-3 text-right font-medium">{copy.maxDrawdown}</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/50">
                  {costRows.map((row) => (
                    <tr key={row.key}>
                      <td className="py-3 pr-3 font-mono text-foreground">{costScenarioLabel(row.key, language)}</td>
                      <td className="px-3 py-3 text-right font-mono text-foreground">{row.sideCostBps ?? '--'}bp</td>
                      <td className="px-3 py-3 text-right font-mono text-[#dc2626]">{pct(row.annualizedReturn)}</td>
                      <td className="py-3 pl-3 text-right font-mono text-[#16a34a]">{pct(row.maxDrawdown)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          <div className="min-w-0">
            <h3 className="text-sm font-semibold text-foreground">{copy.yearly}</h3>
            <div className="mt-3 h-[260px]">
              <ResponsiveContainer width="100%" height="100%" minWidth={0} initialDimension={{ width: 520, height: 260 }}>
                <BarChart data={report.chartData.yearlyReturns} margin={{ top: 10, right: 8, bottom: 4, left: 0 }}>
                  <CartesianGrid stroke={GRID} vertical={false} />
                  <XAxis dataKey="year" tick={{ fontSize: 11 }} />
                  <YAxis tickFormatter={chartPercent} tick={{ fontSize: 11 }} width={48} />
                  <Tooltip formatter={(value) => pct(Number(value))} />
                  <Bar dataKey="strategy" name={copy.strategy} radius={[2, 2, 0, 0]}>
                    {report.chartData.yearlyReturns.map((item) => (
                      <Cell key={item.year} fill={item.strategy >= 0 ? POSITIVE : NEGATIVE} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
        <div className="mt-8 grid gap-8 xl:grid-cols-2">
          <div className="min-w-0">
            <h3 className="text-sm font-semibold text-foreground">{copy.startSensitivity}</h3>
            <div className="mt-3 h-[250px]">
              <ResponsiveContainer width="100%" height="100%" minWidth={0} initialDimension={{ width: 520, height: 250 }}>
                <BarChart data={report.stability.startDateSensitivity} margin={{ top: 10, right: 8, bottom: 4, left: 0 }}>
                  <CartesianGrid stroke={GRID} vertical={false} />
                  <XAxis dataKey="actualStart" tick={{ fontSize: 10 }} minTickGap={20} />
                  <YAxis tickFormatter={chartPercent} tick={{ fontSize: 11 }} width={48} />
                  <Tooltip formatter={(value) => signedPct(Number(value))} />
                  <Bar dataKey="annualizedReturnDelta" name={copy.annualDelta} radius={[2, 2, 0, 0]}>
                    {report.stability.startDateSensitivity.map((item) => (
                      <Cell key={item.actualStart} fill={(item.annualizedReturnDelta ?? 0) >= 0 ? POSITIVE : NEGATIVE} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div className="min-w-0">
            <h3 className="text-sm font-semibold text-foreground">{copy.regime}</h3>
            <div className="mt-3 space-y-3 border-t border-border/60 pt-3">
              {regimeRows.length > 0 ? regimeRows.map((row) => (
                  <div key={row.key} className="grid grid-cols-[1fr_auto_auto] items-center gap-4 text-sm">
                    <span className="font-mono text-foreground">{row.key}</span>
                    <span className={metricColor(row.annualizedReturn)}>{pct(row.annualizedReturn)}</span>
                    <span className="font-mono text-[#16a34a]">DD {pct(row.maxDrawdown)}</span>
                  </div>
                )) : (
                  <p className="text-sm leading-6 text-secondary-text">
                    {regimeAttribution.reason ?? copy.missingBenchmark}
                  </p>
                )}
            </div>
          </div>
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
