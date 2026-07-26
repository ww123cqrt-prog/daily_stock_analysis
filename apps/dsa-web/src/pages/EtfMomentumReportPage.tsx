import type React from 'react';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  AlertTriangle,
  ArrowLeft,
  Check,
  FileDown,
  Loader2,
  Play,
  RefreshCw,
  ShieldAlert,
  X,
} from 'lucide-react';
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
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {
  etfMomentumApi,
  isEtfMomentumStrategyReport,
  type EtfMomentumStrategyResponse,
} from '../api/etfMomentum';
import { quantApi } from '../api/quant';
import { Badge, EmptyState, InlineAlert } from '../components/common';
import { useUiLanguage } from '../contexts/UiLanguageContext';

const POSITIVE = '#dc2626';
const NEGATIVE = '#16a34a';
const BLUE = '#2563eb';
const AMBER = '#d97706';
const GRAY = '#64748b';
const GRID = 'hsl(var(--border) / 0.45)';

const COPY = {
  zh: {
    title: 'ETF 动量轮动完整策略报告',
    eyebrow: 'Fixed-rule research',
    subtitle: '固定比较创业板 ETF（159915）和纳指 ETF（513100）的 20 日动量与 20 日均线，不做参数优化。',
    back: '返回回测页',
    workflow: '工作流',
    reload: '重新读取',
    regenerate: '重新计算报告',
    generating: '生成中',
    print: '打印或导出 PDF',
    unverifiedTitle: '当前数据尚未核验，不可用于交易',
    unverifiedMessage: '报告暂时使用新浪原始价格。513100 在 2022-01-14 存在未解释的价格断点，东财前复权数据和第二源核验尚未完成。',
    staleTitle: '这份报告可能已过期',
    staleMessage: '报告生成时间距现在已超过 72 小时。请先重新生成，再参考最新研究信号。',
    conclusion: '先说结论',
    conclusionTitle: '结果值得继续研究，但还不能当成可执行策略',
    conclusionBody: '收益和成本压力结果有吸引力，但季度稳定性未达到参考线，而且基础价格仍有复权证据缺口。这里展示的是研究证据，不是买卖建议。',
    whatItDoes: '它每天收盘比较两只 ETF 的 20 日涨幅，只在领涨者站上自身 20 日均线时，于下一交易日开盘持有；否则持有现金。',
    latestSignal: '最新研究信号',
    asOf: '信号日期',
    leader: '20 日领先者',
    currentTarget: '下一开盘研究目标',
    signalWarning: '该信号未经过复权数据核验，不可直接交易。',
    ruleTitle: '固定规则',
    buyRule: '买入',
    buyRuleText: '20 日涨幅第一，且收盘价严格高于自身 20 日均线；下一交易日开盘执行。',
    sellRule: '卖出',
    sellRuleText: '持仓不再是涨幅第一，或收盘价跌破自身 20 日均线；下一交易日开盘退出。',
    cashRule: '空仓',
    cashRuleText: '涨幅第一的 ETF 也没有站上自身 20 日均线时，持有 CASH。',
    core: '核心结果',
    baselineAnnual: '20bp 成本年化收益',
    baselineAnnualHelp: '把每次买卖成本计入后，折算成每年的复合收益。',
    totalReturn: '20bp 成本总收益',
    totalReturnHelp: '从回测起点到终点，1 元累计变成多少收益。',
    maxDrawdown: '最大回撤',
    maxDrawdownHelp: '历史上从净值高点到后续低点最深跌幅。',
    stressAnnual: '40bp 压力成本年化',
    stressAnnualHelp: '把交易成本加倍后，策略剩下的年化收益。',
    delayedAnnual: '延迟一天执行年化',
    delayedAnnualHelp: '比原规则再晚一个交易日买卖，检查信号是否过度依赖时点。',
    wealthTitle: '收益与基准',
    wealthBody: '同一起点归一为 1。策略线为红色；创业板 ETF 为蓝色；纳指 ETF 为琥珀色；两只等权为灰色。',
    strategyLine: '策略',
    equalWeight: '两只等权',
    anomaly: '数据异常',
    riskTitle: '风险与回撤',
    drawdown: '回撤曲线',
    drawdownHelp: '绿色越深，表示距离此前净值高点越远。',
    drawdownEpisodes: '最大的 5 次回撤',
    start: '开始',
    trough: '最低点',
    recovery: '恢复',
    days: '交易日',
    notRecovered: '尚未恢复',
    longestUnderwater: '最长未创新高',
    staged: '分阶段表现',
    quarterly: '季度收益',
    yearly: '年度收益',
    startSensitivity: '起始日期敏感性',
    quarterRatios: '三种季度口径必须分开看',
    legacyRatio: '15bp 早期可比口径',
    allRatio: '20bp 含首个可用季度',
    strictRatio: '20bp 严格完整季度',
    stability: '稳定性与压力测试',
    costs: '10 / 20 / 40bp 成本对比',
    delayed: '延迟执行',
    parameterGrid: '15 / 20 / 25 日参数九宫格',
    parameterNote: '这些邻近参数只用于检查稳定性，不能用来事后挑选表现最好的一格。',
    referenceTitle: '七项参考检查',
    referenceNote: '参考线不是自动淘汰线。所有达到和未达到的项目都展示，由你决定是否接受。',
    met: '达到参考线',
    unmet: '未达到参考线',
    holdings: '持仓行为',
    monthlyAllocation: '月度持仓比例',
    switches: '换仓次数',
    averageHold: '平均持仓天数',
    cashRatio: '空仓比例',
    longestCash: '最长空仓期',
    cashMeaning: 'CASH 不是额外标的，而是两只 ETF 都不满足趋势条件时不持仓。',
    gaps: '失效风险与证据缺口',
    failureConditions: '这套规则可能在这些情况下失效',
    evidence: '仍缺少的证据',
    paperTracking: '60 个交易日纸面跟踪',
    paperProgress: '已完成 {done} / 60 个交易日',
    source: '当前价格源',
    adjustment: '复权口径',
    coverage: '东财 qfq 覆盖率',
    generated: '生成时间',
    reproducibility: '复现信息',
    missingTitle: '尚未生成 ETF 策略报告',
    missingDescription: '点击“生成报告”后，后台会读取本地 ETF 数据并生成统一 JSON；不会连接券商或下单。',
    generate: '生成报告',
    invalidTitle: 'ETF 策略报告无法读取',
    invalidDescription: '报告文件损坏或超出安全大小。请重新生成后再试。',
    loading: '正在读取 ETF 策略报告',
    generationFailed: '报告生成失败',
    readFailed: '报告读取失败',
  },
  en: {
    title: 'Complete ETF Momentum Rotation Report',
    eyebrow: 'Fixed-rule research',
    subtitle: 'A fixed 20-day momentum and 20-day moving-average rule for ChiNext ETF 159915 and Nasdaq ETF 513100. No parameter optimization.',
    back: 'Back to backtests',
    workflow: 'Workflow',
    reload: 'Reload',
    regenerate: 'Recalculate report',
    generating: 'Generating',
    print: 'Print or export PDF',
    unverifiedTitle: 'Data is not verified and must not be used for trading',
    unverifiedMessage: 'The report currently uses raw Sina prices. The unexplained 513100 discontinuity on 2022-01-14 remains, and Eastmoney qfq plus second-source verification are incomplete.',
    staleTitle: 'This report may be stale',
    staleMessage: 'It was generated more than 72 hours ago. Regenerate it before reading the latest research signal.',
    conclusion: 'Conclusion first',
    conclusionTitle: 'Worth further research, but not an executable strategy',
    conclusionBody: 'Return and cost-stress results are attractive, but quarterly consistency misses its reference and price-adjustment evidence is incomplete. This is research evidence, not trading advice.',
    whatItDoes: 'At each close, the rule compares 20-day returns. It holds the leader at the next open only when that ETF is above its own 20-day moving average; otherwise it holds cash.',
    latestSignal: 'Latest research signal',
    asOf: 'Signal date',
    leader: '20-day leader',
    currentTarget: 'Next-open research target',
    signalWarning: 'This signal has not passed adjusted-data verification and must not be traded directly.',
    ruleTitle: 'Fixed rules',
    buyRule: 'Buy',
    buyRuleText: 'The 20-day return leader must close strictly above its own 20-day moving average; execute at the next open.',
    sellRule: 'Sell',
    sellRuleText: 'Exit at the next open when the holding loses first rank or closes below its moving average.',
    cashRule: 'Cash',
    cashRuleText: 'Hold CASH when even the return leader is not above its own moving average.',
    core: 'Core results',
    baselineAnnual: 'Annual return after 20bp cost',
    baselineAnnualHelp: 'Compound annual return after applying the baseline trading cost.',
    totalReturn: 'Total return after 20bp cost',
    totalReturnHelp: 'Cumulative return from the first backtest date to the last.',
    maxDrawdown: 'Maximum drawdown',
    maxDrawdownHelp: 'The deepest historical decline from a prior equity high.',
    stressAnnual: 'Annual return at 40bp stress cost',
    stressAnnualHelp: 'Annual return after doubling the assumed trading cost.',
    delayedAnnual: 'Annual return with one-day delay',
    delayedAnnualHelp: 'Checks whether the result depends too heavily on immediate execution.',
    wealthTitle: 'Returns and benchmarks',
    wealthBody: 'All series start at 1. Strategy is red, 159915 blue, 513100 amber, and equal weight gray.',
    strategyLine: 'Strategy',
    equalWeight: 'Equal weight',
    anomaly: 'Data anomaly',
    riskTitle: 'Risk and drawdown',
    drawdown: 'Drawdown curve',
    drawdownHelp: 'Deeper green means the strategy is farther below its previous equity high.',
    drawdownEpisodes: 'Five largest drawdowns',
    start: 'Start',
    trough: 'Trough',
    recovery: 'Recovery',
    days: 'Trading days',
    notRecovered: 'Not recovered',
    longestUnderwater: 'Longest time below high',
    staged: 'Performance by period',
    quarterly: 'Quarterly returns',
    yearly: 'Annual returns',
    startSensitivity: 'Start-date sensitivity',
    quarterRatios: 'Keep the three quarterly definitions separate',
    legacyRatio: 'Earlier 15bp comparison',
    allRatio: '20bp including first available quarter',
    strictRatio: '20bp strict complete quarters',
    stability: 'Stability and stress tests',
    costs: '10 / 20 / 40bp cost comparison',
    delayed: 'Delayed execution',
    parameterGrid: '15 / 20 / 25-day parameter matrix',
    parameterNote: 'Neighboring parameters are diagnostics only; they cannot be used to cherry-pick the best cell.',
    referenceTitle: 'Seven reference checks',
    referenceNote: 'References are advisory, not automatic vetoes. Every pass and miss remains visible for your decision.',
    met: 'Reference met',
    unmet: 'Reference missed',
    holdings: 'Holding behavior',
    monthlyAllocation: 'Monthly allocation share',
    switches: 'Switches',
    averageHold: 'Average holding days',
    cashRatio: 'Cash share',
    longestCash: 'Longest cash period',
    cashMeaning: 'CASH is not another asset. It means neither ETF passes the trend condition.',
    gaps: 'Failure risks and evidence gaps',
    failureConditions: 'Conditions that can break the rule',
    evidence: 'Evidence still missing',
    paperTracking: '60-trading-day paper tracking',
    paperProgress: '{done} / 60 trading days completed',
    source: 'Selected price source',
    adjustment: 'Adjustment basis',
    coverage: 'Eastmoney qfq coverage',
    generated: 'Generated at',
    reproducibility: 'Reproducibility',
    missingTitle: 'ETF strategy report has not been generated',
    missingDescription: 'Generate it from local ETF data. This does not connect to a broker or place orders.',
    generate: 'Generate report',
    invalidTitle: 'ETF strategy report cannot be read',
    invalidDescription: 'The report is malformed or exceeds the safe response size. Regenerate it and retry.',
    loading: 'Loading ETF strategy report',
    generationFailed: 'Report generation failed',
    readFailed: 'Report loading failed',
  },
} as const;

const REFERENCE_LABELS = {
  zh: {
    positive_quarter_ratio: '严格完整季度正收益比例',
    baseline_annualized_return: '20bp 成本年化收益',
    baseline_max_drawdown: '20bp 成本最大回撤',
    stress_annualized_return: '40bp 压力成本年化收益',
    delayed_annualized_return: '延迟一天执行年化收益',
    parameter_positive_ratio: '参数邻域正年化比例',
    parameter_max_drawdown_pass_ratio: '参数邻域回撤不超过 35% 的比例',
  },
  en: {
    positive_quarter_ratio: 'Positive strict-quarter ratio',
    baseline_annualized_return: 'Annual return after 20bp cost',
    baseline_max_drawdown: 'Maximum drawdown after 20bp cost',
    stress_annualized_return: 'Annual return at 40bp stress cost',
    delayed_annualized_return: 'Annual return with one-day delay',
    parameter_positive_ratio: 'Positive-return parameter-neighbor ratio',
    parameter_max_drawdown_pass_ratio: 'Parameter neighbors within 35% drawdown',
  },
} as const;

const EVIDENCE_LABELS = {
  zh: {
    unresolved_price_discontinuity: '513100 在 2022-01-14 的价格断点尚未解释。',
    sixty_trading_day_paper_tracking_not_done: '60 个交易日纸面跟踪尚未完成。',
    independent_source_cross_check_not_done: '独立第二数据源交叉核验尚未完成。',
    dividend_and_split_adjustment_not_verified: '分红、拆分和复权口径尚未核验。',
    dividend_total_return_not_verified: '含分红再投资的总收益口径尚未核验。',
  },
  en: {
    unresolved_price_discontinuity: 'The 513100 price discontinuity on 2022-01-14 is unresolved.',
    sixty_trading_day_paper_tracking_not_done: 'The 60-trading-day paper track is incomplete.',
    independent_source_cross_check_not_done: 'Independent second-source verification is incomplete.',
    dividend_and_split_adjustment_not_verified: 'Dividend, split, and adjustment handling is unverified.',
    dividend_total_return_not_verified: 'Dividend-reinvested total return is unverified.',
  },
} as const;

const FAILURE_CONDITIONS_EN = [
  'Both ETFs can fall together. Cash reduces exposure only after the close signal and cannot remove overnight gaps.',
  'Sideways markets can repeatedly cross the moving average and turn small moves into costs and slippage.',
  'Twenty-day momentum reacts late to fast reversals and can enter near a high before exiting after the decline.',
  'Two growth-oriented ETFs are not broad diversification; shared style, currency, or overseas shocks can hit both.',
  '513100 also carries US market timing, FX, QDII quota, tracking-error, and exchange premium/discount risk.',
];

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

function signalLabel(signal: string): string {
  if (signal === 'SZ159915') return '159915 创业板 ETF';
  if (signal === 'SH513100') return '513100 纳指 ETF';
  return 'CASH';
}

function Section({
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

function MetricBlock({
  label,
  value,
  help,
  tone,
}: {
  label: string;
  value: string;
  help: string;
  tone?: 'positive' | 'negative' | 'neutral';
}) {
  const valueClass = tone === 'positive'
    ? 'text-[#dc2626]'
    : tone === 'negative'
      ? 'text-[#16a34a]'
      : 'text-foreground';
  return (
    <div className="min-h-[132px] border-l-2 border-border/70 px-4 py-2 first:border-l-0 sm:first:border-l-2">
      <p className="text-xs font-medium text-muted-text">{label}</p>
      <p className={`mt-2 font-mono text-2xl font-semibold ${valueClass}`}>{value}</p>
      <p className="mt-2 text-xs leading-5 text-secondary-text">{help}</p>
    </div>
  );
}

function ReturnBarChart({
  data,
  dataKey,
}: {
  data: Array<Record<string, unknown>>;
  dataKey: string;
}) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data} margin={{ top: 12, right: 8, bottom: 4, left: 0 }}>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis dataKey={dataKey} tick={{ fontSize: 11 }} minTickGap={18} />
        <YAxis tickFormatter={chartPercent} tick={{ fontSize: 11 }} width={48} />
        <Tooltip formatter={(value) => pct(Number(value))} />
        <Bar dataKey="return" radius={[2, 2, 0, 0]}>
          {data.map((item, index) => (
            <Cell key={`${String(item[dataKey])}-${index}`} fill={Number(item.return) >= 0 ? POSITIVE : NEGATIVE} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

function isStale(generatedAt: string): boolean {
  const generated = new Date(generatedAt).getTime();
  return Number.isFinite(generated) && Date.now() - generated > 72 * 60 * 60 * 1000;
}

async function waitForJob(jobId: string): Promise<'completed' | 'failed' | 'cancelled'> {
  for (let attempt = 0; attempt < 180; attempt += 1) {
    const job = await quantApi.getJob(jobId);
    if (job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled') {
      return job.status;
    }
    await new Promise((resolve) => window.setTimeout(resolve, 1000));
  }
  return 'failed';
}

const EtfMomentumReportPage: React.FC = () => {
  const { language } = useUiLanguage();
  const copy = COPY[language];
  const [response, setResponse] = useState<EtfMomentumStrategyResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadReport = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setResponse(await etfMomentumApi.getReport());
    } catch {
      setError(copy.readFailed);
    } finally {
      setLoading(false);
    }
  }, [copy.readFailed]);

  useEffect(() => {
    document.title = `${copy.title} - DSA`;
    void loadReport();
  }, [copy.title, loadReport]);

  const generateReport = useCallback(async () => {
    if (generating) return;
    setGenerating(true);
    setError(null);
    try {
      const job = await quantApi.submitJob('generate_etf_momentum_report');
      const status = job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled'
        ? job.status
        : await waitForJob(job.jobId);
      if (status !== 'completed') {
        throw new Error(status);
      }
      await loadReport();
    } catch {
      setError(copy.generationFailed);
    } finally {
      setGenerating(false);
    }
  }, [copy.generationFailed, generating, loadReport]);

  const report = isEtfMomentumStrategyReport(response) ? response : null;
  const stale = report ? isStale(report.generatedAt) : false;
  const failureConditions = useMemo(() => {
    if (!report) return [];
    return language === 'en'
      ? report.risk.knownFailureConditions.map((_, index) => FAILURE_CONDITIONS_EN[index] ?? report.risk.knownFailureConditions[index])
      : report.risk.knownFailureConditions;
  }, [language, report]);

  const header = (
    <header className="flex flex-col gap-5 border-b border-border/60 pb-6 lg:flex-row lg:items-end lg:justify-between">
      <div className="min-w-0">
        <span className="label-uppercase">{copy.eyebrow}</span>
        <h1 className="mt-2 text-2xl font-semibold text-foreground sm:text-3xl">{copy.title}</h1>
        <p className="mt-2 max-w-4xl text-sm leading-6 text-secondary-text">{copy.subtitle}</p>
      </div>
      <div className="flex flex-wrap gap-2" data-print-hide>
        <Link to="/backtest" className="btn-secondary h-9 gap-2 px-3 text-sm">
          <ArrowLeft className="h-4 w-4" />
          {copy.back}
        </Link>
        <Link to="/workflow" className="btn-secondary h-9 px-3 text-sm">{copy.workflow}</Link>
        <button type="button" className="btn-secondary h-9 gap-2 px-3 text-sm" onClick={() => void loadReport()} disabled={loading}>
          <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          {copy.reload}
        </button>
        <button type="button" className="btn-primary h-9 gap-2 px-3 text-sm" onClick={() => void generateReport()} disabled={generating}>
          {generating ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
          {generating ? copy.generating : copy.regenerate}
        </button>
        <button type="button" className="btn-secondary h-9 gap-2 px-3 text-sm" onClick={() => window.print()}>
          <FileDown className="h-4 w-4" />
          {copy.print}
        </button>
      </div>
    </header>
  );

  if (loading && !response) {
    return (
      <main className="etf-report-page min-h-screen bg-base px-4 py-6 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-[1480px]">
          {header}
          <div className="flex min-h-[420px] items-center justify-center gap-3 text-sm text-secondary-text">
            <Loader2 className="h-5 w-5 animate-spin" />
            {copy.loading}
          </div>
        </div>
      </main>
    );
  }

  if (!report) {
    const missing = response?.status === 'missing';
    return (
      <main className="etf-report-page min-h-screen bg-base px-4 py-6 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-[1480px]">
          {header}
          {error ? <InlineAlert className="mt-6" variant="danger" title={copy.readFailed} message={error} /> : null}
          <div className="py-10">
            <EmptyState
              icon={missing ? <FileDown className="h-8 w-8" /> : <AlertTriangle className="h-8 w-8" />}
              title={missing ? copy.missingTitle : copy.invalidTitle}
              description={missing ? copy.missingDescription : copy.invalidDescription}
              action={(
                <button type="button" className="btn-primary gap-2" onClick={() => void generateReport()} disabled={generating}>
                  {generating ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                  {generating ? copy.generating : copy.generate}
                </button>
              )}
            />
          </div>
        </div>
      </main>
    );
  }

  const baseline = report.performance.variants.baseline?.metrics ?? {};
  const stress = report.performance.variants.stress?.metrics ?? {};
  const quarter = report.stability.quarterConsistency;
  const cash = report.risk.cashMetrics.signalCash ?? {};

  return (
    <main className="etf-report-page min-h-screen bg-base px-4 py-6 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-[1480px]">
        {header}
        <div className="mt-5 space-y-3">
          {report.summary.dataStatus !== 'verified_qfq' ? (
            <InlineAlert
              variant="danger"
              title={copy.unverifiedTitle}
              message={copy.unverifiedMessage}
            />
          ) : null}
          {stale ? <InlineAlert variant="warning" title={copy.staleTitle} message={copy.staleMessage} /> : null}
          {error ? <InlineAlert variant="danger" title={copy.generationFailed} message={error} /> : null}
        </div>

        <Section title={copy.conclusion}>
          <div className="grid gap-6 lg:grid-cols-[1.4fr_0.8fr]">
            <div className="max-w-4xl">
              <h3 className="text-lg font-semibold text-foreground">{copy.conclusionTitle}</h3>
              <p className="mt-3 text-sm leading-7 text-secondary-text">{copy.conclusionBody}</p>
              <p className="mt-3 text-sm leading-7 text-secondary-text">{copy.whatItDoes}</p>
            </div>
            <div className="border-l-2 border-border/70 pl-5">
              <div className="flex flex-wrap items-center gap-2">
                <p className="text-sm font-semibold text-foreground">{copy.latestSignal}</p>
                <Badge variant="danger">{copy.unverifiedTitle.split('，')[1] ?? copy.unverifiedTitle}</Badge>
              </div>
              <dl className="mt-4 grid grid-cols-[auto_1fr] gap-x-4 gap-y-2 text-sm">
                <dt className="text-muted-text">{copy.asOf}</dt><dd className="font-mono text-foreground">{report.latestSignal.asOfDate}</dd>
                <dt className="text-muted-text">{copy.leader}</dt><dd className="font-medium text-foreground">{signalLabel(report.latestSignal.leader)}</dd>
                <dt className="text-muted-text">{copy.currentTarget}</dt><dd className="font-semibold text-foreground">{signalLabel(report.latestSignal.nextOpenResearchTarget)}</dd>
              </dl>
              <p className="mt-3 text-xs leading-5 text-[#16a34a]">{copy.signalWarning}</p>
            </div>
          </div>
          <div className="mt-7 grid gap-4 md:grid-cols-3">
            {[
              [copy.buyRule, copy.buyRuleText],
              [copy.sellRule, copy.sellRuleText],
              [copy.cashRule, copy.cashRuleText],
            ].map(([label, text]) => (
              <div key={label} className="border-t-2 border-border/70 pt-3">
                <p className="text-sm font-semibold text-foreground">{label}</p>
                <p className="mt-2 text-sm leading-6 text-secondary-text">{text}</p>
              </div>
            ))}
          </div>
        </Section>

        <Section title={copy.core}>
          <div className="grid gap-y-5 sm:grid-cols-2 xl:grid-cols-5">
            <MetricBlock label={copy.baselineAnnual} value={pct(baseline.annualizedReturn)} help={copy.baselineAnnualHelp} tone="positive" />
            <MetricBlock label={copy.totalReturn} value={pct(baseline.totalReturn)} help={copy.totalReturnHelp} tone="positive" />
            <MetricBlock label={copy.maxDrawdown} value={pct(baseline.maxDrawdown)} help={copy.maxDrawdownHelp} tone="negative" />
            <MetricBlock label={copy.stressAnnual} value={pct(stress.annualizedReturn)} help={copy.stressAnnualHelp} tone="positive" />
            <MetricBlock label={copy.delayedAnnual} value={pct(report.performance.delayedExecution.metrics.annualizedReturn)} help={copy.delayedAnnualHelp} tone="positive" />
          </div>
        </Section>

        <Section title={copy.wealthTitle} description={copy.wealthBody}>
          <div className="h-[360px] w-full" data-testid="wealth-chart">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={report.chartData.wealth} margin={{ top: 16, right: 16, bottom: 8, left: 0 }}>
                <CartesianGrid stroke={GRID} vertical={false} />
                <XAxis dataKey="date" tick={{ fontSize: 11 }} minTickGap={54} />
                <YAxis tick={{ fontSize: 11 }} width={54} />
                <Tooltip formatter={(value) => number(Number(value))} />
                <Legend />
                <Line type="monotone" dataKey="strategy" name={copy.strategyLine} stroke={POSITIVE} strokeWidth={2.4} dot={false} />
                <Line type="monotone" dataKey="SZ159915" name="159915" stroke={BLUE} strokeWidth={1.5} dot={false} />
                <Line type="monotone" dataKey="SH513100" name="513100" stroke={AMBER} strokeWidth={1.5} dot={false} />
                <Line type="monotone" dataKey="equalWeight" name={copy.equalWeight} stroke={GRAY} strokeWidth={1.4} dot={false} />
                {report.chartData.anomalyDates.map((date) => (
                  <ReferenceLine key={date} x={date} stroke={NEGATIVE} strokeDasharray="4 4" label={`${copy.anomaly} ${date}`} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
          {report.chartData.anomalyDates.length ? (
            <div className="mt-3 flex flex-wrap gap-2 text-xs text-secondary-text">
              {report.chartData.anomalyDates.map((date) => <span key={date}>{copy.anomaly}: <strong className="font-mono text-foreground">{date}</strong></span>)}
            </div>
          ) : null}
        </Section>

        <Section title={copy.riskTitle}>
          <div className="grid gap-8 xl:grid-cols-[1.35fr_1fr]">
            <div>
              <h3 className="text-sm font-semibold text-foreground">{copy.drawdown}</h3>
              <p className="mt-1 text-xs text-secondary-text">{copy.drawdownHelp}</p>
              <div className="mt-3 h-[300px]" data-testid="drawdown-chart">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={report.chartData.drawdown} margin={{ top: 10, right: 10, bottom: 4, left: 0 }}>
                    <CartesianGrid stroke={GRID} vertical={false} />
                    <XAxis dataKey="date" tick={{ fontSize: 11 }} minTickGap={54} />
                    <YAxis tickFormatter={chartPercent} tick={{ fontSize: 11 }} width={52} />
                    <Tooltip formatter={(value) => pct(Number(value))} />
                    <Area type="monotone" dataKey="drawdown" stroke={NEGATIVE} fill={NEGATIVE} fillOpacity={0.18} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
            <div className="min-w-0">
              <div className="flex items-center justify-between gap-3">
                <h3 className="text-sm font-semibold text-foreground">{copy.drawdownEpisodes}</h3>
                <span className="text-xs text-secondary-text">{copy.longestUnderwater}: {report.risk.longestUnderwaterTradingDays} {copy.days}</span>
              </div>
              <div className="mt-3 overflow-x-auto">
                <table className="min-w-full text-left text-xs">
                  <thead className="border-b border-border/60 text-muted-text"><tr><th className="py-2 pr-3">{copy.start}</th><th className="py-2 pr-3">{copy.trough}</th><th className="py-2 pr-3">{copy.recovery}</th><th className="py-2 pr-3">{copy.maxDrawdown}</th><th className="py-2">{copy.days}</th></tr></thead>
                  <tbody className="divide-y divide-border/40">
                    {report.risk.drawdownEpisodes.map((item) => (
                      <tr key={`${item.startDate}-${item.troughDate}`}>
                        <td className="py-2.5 pr-3 font-mono text-foreground">{item.startDate}</td>
                        <td className="py-2.5 pr-3 font-mono text-foreground">{item.troughDate}</td>
                        <td className="py-2.5 pr-3 font-mono text-secondary-text">{item.recoveryDate ?? copy.notRecovered}</td>
                        <td className="py-2.5 pr-3 font-mono text-[#16a34a]">{pct(item.maxDrawdown)}</td>
                        <td className="py-2.5 font-mono text-foreground">{item.tradingDays}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </Section>

        <Section title={copy.staged}>
          <div className="grid gap-7 xl:grid-cols-2">
            <div>
              <h3 className="text-sm font-semibold text-foreground">{copy.quarterly}</h3>
              <div className="mt-3 h-[280px]" data-testid="quarterly-chart">
                <ReturnBarChart data={report.chartData.quarterlyReturns} dataKey="period" />
              </div>
            </div>
            <div>
              <h3 className="text-sm font-semibold text-foreground">{copy.yearly}</h3>
              <div className="mt-3 h-[280px]" data-testid="yearly-chart">
                <ReturnBarChart data={report.chartData.yearlyReturns} dataKey="period" />
              </div>
            </div>
          </div>
          <div className="mt-7 grid gap-4 md:grid-cols-3">
            {[
              [copy.legacyRatio, quarter.earlierComparisonPositiveRatio, `${quarter.earlierComparisonPositiveCount}/${quarter.earlierComparisonCount}`],
              [copy.allRatio, quarter.allAvailablePositiveRatio, `${quarter.allAvailablePositiveCount}/${quarter.allAvailableCount}`],
              [copy.strictRatio, quarter.strictCompletePositiveRatio, `${quarter.strictCompletePositiveCount}/${quarter.strictCompleteCount}`],
            ].map(([label, value, count]) => (
              <div key={String(label)} className="border-l-2 border-border/70 pl-4">
                <p className="text-xs text-muted-text">{label}</p>
                <p className="mt-2 font-mono text-2xl font-semibold text-[#dc2626]">{pct(Number(value))}</p>
                <p className="mt-1 text-xs text-secondary-text">{count}</p>
              </div>
            ))}
          </div>
          <div className="mt-8">
            <h3 className="text-sm font-semibold text-foreground">{copy.startSensitivity}</h3>
            <div className="mt-3 h-[260px]" data-testid="start-sensitivity-chart">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={report.chartData.startDateSensitivity} margin={{ top: 10, right: 10, bottom: 4, left: 0 }}>
                  <CartesianGrid stroke={GRID} vertical={false} />
                  <XAxis dataKey="startDate" tick={{ fontSize: 11 }} />
                  <YAxis tickFormatter={chartPercent} tick={{ fontSize: 11 }} width={50} />
                  <Tooltip formatter={(value) => pct(Number(value))} />
                  <Bar dataKey="annualizedReturn" fill={POSITIVE} radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </Section>

        <Section title={copy.stability} description={copy.referenceNote}>
          <div className="grid gap-8 xl:grid-cols-[0.9fr_1.1fr]">
            <div>
              <h3 className="text-sm font-semibold text-foreground">{copy.costs}</h3>
              <div className="mt-3 h-[250px]" data-testid="cost-chart">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={report.chartData.costStress} margin={{ top: 10, right: 10, bottom: 4, left: 0 }}>
                    <CartesianGrid stroke={GRID} vertical={false} />
                    <XAxis dataKey="label" tick={{ fontSize: 11 }} />
                    <YAxis tickFormatter={chartPercent} tick={{ fontSize: 11 }} width={50} />
                    <Tooltip formatter={(value) => pct(Number(value))} />
                    <Bar dataKey="annualizedReturn" fill={POSITIVE} radius={[2, 2, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-5 border-l-2 border-border/70 pl-4">
                <p className="text-xs text-muted-text">{copy.delayed}</p>
                <p className="mt-1 font-mono text-xl font-semibold text-[#dc2626]">{pct(report.performance.delayedExecution.metrics.annualizedReturn)}</p>
              </div>
            </div>
            <div>
              <h3 className="text-sm font-semibold text-foreground">{copy.parameterGrid}</h3>
              <p className="mt-1 text-xs leading-5 text-secondary-text">{copy.parameterNote}</p>
              <div className="mt-4 grid grid-cols-3 gap-px overflow-hidden border border-border/60 bg-border/60" data-testid="parameter-grid">
                {report.stability.parameterGrid.variants.map((item) => {
                  const primary = item.lookbackDays === report.stability.parameterGrid.primary.lookbackDays
                    && item.movingAverageDays === report.stability.parameterGrid.primary.movingAverageDays;
                  return (
                    <div key={`${item.lookbackDays}-${item.movingAverageDays}`} className={`min-h-[92px] bg-base p-3 ${primary ? 'ring-2 ring-inset ring-[#dc2626]' : ''}`}>
                      <p className="text-xs text-muted-text">{item.lookbackDays} / {item.movingAverageDays}</p>
                      <p className={`mt-2 font-mono text-lg font-semibold ${metricColor(item.metrics.annualizedReturn)}`}>{pct(item.metrics.annualizedReturn)}</p>
                      <p className="mt-1 font-mono text-xs text-[#16a34a]">DD {pct(item.metrics.maxDrawdown)}</p>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
          <div className="mt-8">
            <h3 className="text-sm font-semibold text-foreground">{copy.referenceTitle}</h3>
            <div className="mt-3 grid gap-px overflow-hidden border border-border/60 bg-border/60 md:grid-cols-2 xl:grid-cols-3">
              {report.stability.referenceChecks.map((check) => (
                <div key={check.id} data-testid="reference-check" className="flex min-h-[92px] items-start gap-3 bg-base p-4">
                  {check.met ? <Check className="mt-0.5 h-4 w-4 shrink-0 text-[#dc2626]" /> : <X className="mt-0.5 h-4 w-4 shrink-0 text-[#16a34a]" />}
                  <div className="min-w-0">
                    <p className="text-sm font-medium text-foreground">{REFERENCE_LABELS[language][check.id as keyof typeof REFERENCE_LABELS.zh] ?? check.id}</p>
                    <p className={`mt-1 font-mono text-sm ${check.met ? 'text-[#dc2626]' : 'text-[#16a34a]'}`}>{pct(check.value)} {check.operator} {pct(check.threshold)}</p>
                    <p className="mt-1 text-xs text-secondary-text">{check.met ? copy.met : copy.unmet}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </Section>

        <Section title={copy.holdings} description={copy.cashMeaning}>
          <div className="grid gap-8 xl:grid-cols-[1.4fr_0.8fr]">
            <div className="h-[300px]" data-testid="allocation-chart">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={report.chartData.monthlyAllocations} margin={{ top: 10, right: 10, bottom: 4, left: 0 }}>
                  <CartesianGrid stroke={GRID} vertical={false} />
                  <XAxis dataKey="month" tick={{ fontSize: 11 }} minTickGap={30} />
                  <YAxis domain={[0, 1]} tickFormatter={chartPercent} tick={{ fontSize: 11 }} width={50} />
                  <Tooltip formatter={(value) => pct(Number(value))} />
                  <Legend />
                  <Bar dataKey="SZ159915" name="159915" stackId="allocation" fill={BLUE} />
                  <Bar dataKey="SH513100" name="513100" stackId="allocation" fill={AMBER} />
                  <Bar dataKey="CASH" name="CASH" stackId="allocation" fill={GRAY} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="grid grid-cols-2 gap-4 content-start">
              {[
                [copy.switches, String(baseline.switchCount ?? '--')],
                [copy.averageHold, `${number(report.risk.holdingSummary.averageTradingDays)} ${copy.days}`],
                [copy.cashRatio, pct(cash.cashFraction)],
                [copy.longestCash, `${cash.longestCashStreakDays ?? '--'} ${copy.days}`],
              ].map(([label, value]) => (
                <div key={label} className="border-l-2 border-border/70 pl-4 py-1">
                  <p className="text-xs text-muted-text">{label}</p>
                  <p className="mt-2 font-mono text-lg font-semibold text-foreground">{value}</p>
                </div>
              ))}
            </div>
          </div>
        </Section>

        <Section title={copy.gaps}>
          <div className="grid gap-8 lg:grid-cols-2">
            <div>
              <h3 className="flex items-center gap-2 text-sm font-semibold text-foreground"><ShieldAlert className="h-4 w-4 text-[#16a34a]" />{copy.failureConditions}</h3>
              <ol className="mt-4 space-y-3">
                {failureConditions.map((condition, index) => (
                  <li key={`${index}-${condition}`} className="flex gap-3 text-sm leading-6 text-secondary-text"><span className="font-mono text-muted-text">{String(index + 1).padStart(2, '0')}</span><span>{condition}</span></li>
                ))}
              </ol>
            </div>
            <div>
              <h3 className="flex items-center gap-2 text-sm font-semibold text-foreground"><AlertTriangle className="h-4 w-4 text-[#16a34a]" />{copy.evidence}</h3>
              <div className="mt-4 space-y-3">
                {report.dataQuality.evidenceGaps.map((gap) => (
                  <div key={gap} data-testid="evidence-gap" className="border-l-2 border-[#16a34a] pl-3 text-sm leading-6 text-secondary-text">
                    {EVIDENCE_LABELS[language][gap as keyof typeof EVIDENCE_LABELS.zh] ?? gap}
                  </div>
                ))}
              </div>
              <div className="mt-6 grid gap-3 sm:grid-cols-2">
                <div className="border-t border-border/60 pt-3"><p className="text-xs text-muted-text">{copy.paperTracking}</p><p className="mt-1 text-sm font-medium text-foreground">{copy.paperProgress.replace('{done}', String(report.dataQuality.paperTrackingDaysCompleted))}</p></div>
                <div className="border-t border-border/60 pt-3"><p className="text-xs text-muted-text">{copy.coverage}</p><p className="mt-1 font-mono text-sm font-medium text-foreground">{pct(report.dataQuality.coverageRatio)}</p></div>
                <div className="border-t border-border/60 pt-3"><p className="text-xs text-muted-text">{copy.source}</p><p className="mt-1 break-all font-mono text-xs text-foreground">{report.dataQuality.selectedSource}</p></div>
                <div className="border-t border-border/60 pt-3"><p className="text-xs text-muted-text">{copy.adjustment}</p><p className="mt-1 font-mono text-sm text-foreground">{report.dataQuality.selectedAdjustment}</p></div>
              </div>
            </div>
          </div>
        </Section>

        <footer className="border-t border-border/60 py-6 text-xs leading-5 text-muted-text">
          <p>{copy.generated}: {new Date(report.generatedAt).toLocaleString(language === 'zh' ? 'zh-CN' : 'en-US')}</p>
          <p className="mt-1">{copy.reproducibility}: config {report.reproducibility.configHash?.slice(0, 12) ?? '--'} / data {report.reproducibility.dataManifestHash?.slice(0, 12) ?? '--'} / git {report.reproducibility.gitCommit?.slice(0, 12) ?? '--'}</p>
        </footer>
      </div>
    </main>
  );
};

export default EtfMomentumReportPage;
