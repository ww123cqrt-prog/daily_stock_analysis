import type React from 'react';
import { Fragment, useCallback, useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import type { LucideIcon } from 'lucide-react';
import {
  ArrowRight,
  ArrowUpRight,
  BadgeCheck,
  BarChart3,
  Bell,
  BrainCircuit,
  Building2,
  Archive,
  Database,
  FlaskConical,
  Globe2,
  Landmark,
  LineChart,
  MonitorSmartphone,
  Network,
  PieChart,
  Radio,
  RefreshCw,
  Search,
  ShieldAlert,
  Smartphone,
  TrendingUp,
  Trash2,
  WalletCards,
} from 'lucide-react';
import { quantApi, type QuantJob } from '../api/quant';
import { researchApi } from '../api/research';
import type {
  ResearchChinaAstockReuseItem,
  ResearchAcceptanceChecklistItem,
  ResearchCapabilityMatrixItem,
  ResearchDataSource,
  ResearchExperimentRun,
  ResearchExternalTools,
  ResearchFunctionInventoryItem,
  ResearchStatus,
} from '../api/research';
import { AppPage, Badge, Button, Card, ConfirmDialog, PageHeader, StatusDot } from '../components/common';
import { useUiLanguage } from '../contexts/UiLanguageContext';
import { resolveMlflowUiUrl } from '../utils/mlflowUiUrl';

type StatusTone = 'success' | 'warning' | 'danger' | 'info' | 'neutral';
type BadgeVariant = 'default' | 'success' | 'warning' | 'danger' | 'info' | 'history';
type SupportState = 'available' | 'partial' | 'future' | 'read_only' | 'deferred' | 'not_planned';

type StatusItem = {
  label: string;
  value: string;
  detail: string;
  tone: StatusTone;
};

type CapabilityItem = {
  title: string;
  status: SupportState;
  description: string;
  focus: string[];
  route: string;
  routeLabel: string;
  icon: LucideIcon;
};

type ModeItem = {
  title: string;
  status: SupportState;
  icon: LucideIcon;
  description: string;
  points: string[];
};

type EtfBoundaryItem = {
  title: string;
  status: SupportState;
  detail: string;
};

type ProjectItem = {
  name: string;
  role: string;
  detail: string;
  icon: LucideIcon;
};

type SelfServiceCoverageItem = {
  module: string;
  status: SupportState;
  entry: string;
  route?: string;
  tests: string;
  browser: string;
  review: string;
  next: string;
  restrictions?: string[];
};

type FunctionInventoryItem = ResearchFunctionInventoryItem;
type ChinaAstockReuseItem = ResearchChinaAstockReuseItem;
type AcceptanceChecklistItem = ResearchAcceptanceChecklistItem;

type DevelopmentQueueItem = {
  priority: string;
  title: string;
  status: SupportState;
  entry: string;
  route?: string;
  next: string;
  verification: string;
};

type FutureItem = {
  title: string;
  status: string;
  description: string;
  icon: LucideIcon;
};

type ExperimentComparison = {
  cleanedRun: ResearchExperimentRun;
  canonicalRun: ResearchExperimentRun;
  metricKey: string;
  cleanedValue: number;
  canonicalValue: number;
  delta: number;
};

const DEFAULT_MLFLOW_URL = (import.meta.env.VITE_MLFLOW_URL as string) || 'http://127.0.0.1:6006/';
const CATDESK_URL = (import.meta.env.VITE_CATDESK_URL as string) || 'http://127.0.0.1:8765/';
const COMPARISON_METRIC_KEY = '1day.excess_return_with_cost.annualized_return';

const STATUS_STYLE: Record<SupportState, { zh: string; en: string; variant: BadgeVariant; tone: StatusTone }> = {
  available: { zh: '可用', en: 'Available', variant: 'success', tone: 'success' },
  partial: { zh: '部分可用', en: 'Partial', variant: 'warning', tone: 'warning' },
  future: { zh: '待补强', en: 'Planned', variant: 'default', tone: 'neutral' },
  read_only: { zh: '只读可用', en: 'Read-only', variant: 'info', tone: 'info' },
  deferred: { zh: '暂缓', en: 'Deferred', variant: 'default', tone: 'neutral' },
  not_planned: { zh: '不计划', en: 'Not planned', variant: 'danger', tone: 'danger' },
};

const DATA_STATUS_STYLE: Record<string, { zh: string; en: string; variant: BadgeVariant; tone: StatusTone }> = {
  available: { zh: '可用', en: 'Available', variant: 'success', tone: 'success' },
  linked: { zh: '已链接', en: 'Linked', variant: 'info', tone: 'info' },
  invalid: { zh: '配置无效', en: 'Invalid', variant: 'danger', tone: 'danger' },
  partial: { zh: '部分可用', en: 'Partial', variant: 'warning', tone: 'warning' },
  missing: { zh: '缺失', en: 'Missing', variant: 'danger', tone: 'danger' },
  unknown: { zh: '未知', en: 'Unknown', variant: 'default', tone: 'neutral' },
};

const RESEARCH_TEXT = {
  zh: {
    documentTitle: '研究工作台 - DSA',
    eyebrow: '本地投研入口',
    title: '研究工作台',
    description: '把股票、场内基金/ETF、量化实验和后续外部工具接入整理到一个本地投研入口。',
    openWorkflow: '打开工作流',
    openMlflow: '打开 MLflow',
    backHome: '回到首页',
    statusTitle: '服务状态',
    dataStatusTitle: '数据状态',
    dataStatusDescription: '由 quant_workflow 生成的本地数据质量与训练状态，DSA 只读取结果产物。',
    dataHygieneTitle: '数据治理',
    dataHygieneDescription: '本地数据盘点、清理计划、质量报告和维护锁状态。',
    rescanDataHygiene: '重新扫描',
    submittingDataHygiene: '正在提交',
    scanningDataHygiene: '扫描中',
    dataHygieneUpdated: '已更新',
    dataHygieneFailed: '重新扫描失败',
    dataHygieneErrorSeparator: '：',
    dataHygieneCancelled: '任务已取消',
    maintenanceLockAction: '维护锁已锁定',
    maintenanceLockLabel: '维护锁',
    locked: '已锁定',
    notLocked: '未锁定',
    inventoryLabel: '数据盘点',
    cleanupPlanLabel: '清理计划',
    qualityReportLabel: '质量报告',
    cleaningStrategyLabel: '清洗策略',
    manifestLabel: '数据 Manifest',
    readmeLabel: '治理说明',
    officialLabel: '正式',
    candidateLabel: '候选',
    legacyLabel: '旧数据',
    archiveLabel: '待归档',
    actionsLabel: '动作',
    dryRunLabel: 'Dry-run',
    viewReport: '查看报告',
    viewDetails: '查看详情',
    qualitySummaryTitle: '质量摘要',
    cleanupActionsTitle: '清理动作预览',
    actionTypeLabel: '类型',
    sourceLabel: '源路径',
    destinationLabel: '目标路径',
    reasonLabel: '原因',
    riskLabel: '风险',
    confirmationRequired: '需要确认',
    stockUniverse: '股票池',
    usableInstruments: '可用股票',
    totalRows: '总行数',
    dateRange: '日期范围',
    trainable: '可训练',
    notTrainable: '暂不可训练',
    generatedAt: '生成时间',
    qlibProviderTitle: 'Qlib 数据集',
    latestTrainingTitle: '最近训练结果',
    noTrainingResult: '还没有训练结果',
    experimentNotebookTitle: '实验记录本',
    experimentNotebookDescription: '把每次量化训练的动机、数据版本、核心指标、结论和下一步沉淀成中文研究记录。',
    latestNoteTitle: '最新中文记录',
    latestRunsTitle: '最近实验',
    coverageTitle: '能力验收矩阵',
    coverageDescription: '按模块核对“能否在网页端自助完成”和当前验收口径；测试、浏览器、opencode 和安全边界都作为继续推进的门槛。',
    coverageModuleLabel: '模块',
    coverageEntryLabel: '网页入口',
    coverageTestsLabel: '测试脚本',
    coverageBrowserLabel: '浏览器执行',
    coverageReviewLabel: 'opencode复审',
    coverageNextLabel: '下一步',
    inventoryTitle: '功能入口盘点',
    inventoryDescription: '把 DSA 原有页面、量化流水线、Qlib 接入、实验记录和外部观察层统一列账；核心流程必须有网页入口或明确缺口。',
    inventoryAreaLabel: '类别',
    inventoryOwnerLabel: '归属',
    inventoryCoverageLabel: '当前覆盖',
    inventoryGapsLabel: '缺口',
    inventoryVerificationLabel: '验收',
    chinaReuseTitle: 'china-astock-quant 复用清单',
    chinaReuseDescription: '只复用适合研究和回测的能力；交易、券商和自动下单相关模块保留为参考，不进入本阶段主工作台。',
    chinaReuseCapabilityLabel: '能力',
    chinaReuseSourceLabel: '来源',
    chinaReuseDecisionLabel: '决策',
    chinaReuseTargetLabel: '接入目标',
    chinaReuseRestrictionLabel: '限制',
    acceptanceTitle: '阶段验收清单',
    acceptanceDescription: '把当前阶段是否真的可交付列成页面内清单，避免只靠口头说明。',
    acceptanceEvidenceLabel: '证据',
    acceptanceNextLabel: '下一步',
    developmentQueueTitle: '后续开发队列',
    developmentQueueDescription: '把接下来要做的事拆成可验收模块；每一条都必须走测试、浏览器执行和只读复审。',
    developmentPriorityLabel: '优先级',
    developmentEntryLabel: '网页入口',
    developmentActionLabel: '下一步动作',
    developmentVerificationLabel: '验收口径',
    experimentComparisonTitle: '清洗对比',
    experimentComparisonDescription: '同一 baseline 在原生 V2 数据和旧版清洗数据上的核心回测指标。',
    cleanedDataLabel: '清洗后数据',
    canonicalDataLabel: '原生 V2 数据',
    comparisonMetricLabel: '扣成本年化超额收益',
    comparisonDeltaLabel: '差值',
    comparisonNoData: '需要至少一条 cleaned 和一条 canonical 实验记录后才能生成对比。',
    noExperimentNote: '还没有中文实验记录。可以先运行一次训练，再把结论写入 experiment_log.md。',
    noExperimentRuns: '还没有结构化实验条目。',
    archiveExperiment: '归档',
    archiveExperimentAria: '归档实验',
    permanentDeleteExperiment: '永久删除',
    permanentDeleteExperimentAria: '永久删除实验',
    permanentDeleteConfirm: '确认永久删除这条实验记录？这会从 experiment_runs.jsonl 移除，并删除 experiment_log.md 中带 run_id 标记的段落；不会删除 MLflow 和训练文件。',
    deleteExperimentFailed: '操作实验记录失败',
    runIdLabel: 'Run',
    modelLabel: '模型',
    strategyLabel: '策略',
    datasetLabel: '数据区间',
    conclusionLabel: '结论',
    nextStepLabel: '下一步',
    notebookFileLabel: '记录文件',
    statusLoadFailed: '状态读取失败',
    filesLabel: '文件',
    rowsLabel: '行数',
    instrumentsLabel: '标的',
    calendarLabel: '交易日',
    capabilitiesTitle: '六类分析能力',
    capabilitiesDescription: '首版作为统一入口和能力边界说明；实际分析仍复用现有 DSA、Qlib 与数据链路。',
    dualModeTitle: '股票与场内基金/ETF双模式',
    dualModeDescription: '同一套入口下区分研究对象，ETF 不再被当作普通公司股票处理。',
    etfBoundaryTitle: '场内基金/ETF首版支持边界',
    projectMapTitle: '项目关系图',
    projectMapDescription: '各项目保留原职责，DSA 只做本地研究入口和结果聚合。',
    futureTitle: '后续升级项',
    futureDescription: '这些能力本轮只作为路线图展示，默认只读观察，不读取密码、不自动下单。',
    externalToolsTitle: '外部研究工具',
    externalToolsDescription: '先通过跳转把可用工具接到同一个工作台，后续再评估数据和模块是否合并。',
    catdeskTitle: 'CatDesk 9 / 美股催化 A股早报',
    catdeskDescription: '跨市场催化观察：用美股隔夜板块和主题变化生成 A 股候选清单；其中的回测证据是候选信号历史验证，不等同 Qlib 策略回测。',
    catdeskOpen: '打开 CatDesk 9',
    catdeskNote: '当前仅跳转，数据暂不互通；后续再评估重复模块如何合并简化。',
    mlflowNote: 'MLflow 保持为实验记录页，不嵌入或改造；是否已启动以工作流页顶部状态为准。',
    portNote: 'DSA 跟随当前访问地址；本机 8000 可能被其他服务占用。',
    statusItems: [
      { label: 'DSA 服务', value: '当前页面', detail: '本地 WebUI', tone: 'success' },
      { label: '数据源', value: 'AkShare/JQData/本地数据', detail: '按现有链路获取', tone: 'info' },
      { label: 'MLflow', value: '实验记录', detail: '/experiments/（状态见工作流页）', tone: 'info' },
      { label: '端口', value: '跟随当前访问地址', detail: '不硬编码 8000', tone: 'warning' },
    ] satisfies StatusItem[],
    capabilities: [
      {
        title: '基本面分析',
        status: 'partial',
        description: '股票可看财报、盈利、估值、行业和竞争力；ETF 基本面降级为标的指数与主题逻辑。',
        focus: ['财报/盈利/估值', '行业与竞争力', 'ETF 按主题和指数降级'],
        route: '/',
        routeLabel: '去首页分析',
        icon: Building2,
      },
      {
        title: '技术面/技术分析',
        status: 'available',
        description: '复用价格序列、均线、MACD、RSI/KDJ、成交量与形态判断。',
        focus: ['K线与均线', 'MACD/RSI/KDJ', '量价结构'],
        route: '/alerts',
        routeLabel: '配置技术告警',
        icon: LineChart,
      },
      {
        title: '资金面分析',
        status: 'partial',
        description: '股票侧可结合资金流、融资融券、龙虎榜；ETF 侧优先看成交额、换手和流动性。',
        focus: ['主力/北向资金', '融资融券/龙虎榜', 'ETF 成交额与流动性'],
        route: '/',
        routeLabel: '查看综合分析',
        icon: WalletCards,
      },
      {
        title: '情绪面/情绪分析',
        status: 'partial',
        description: '围绕市场心理、热点、新闻和涨跌家数做辅助判断，短线权重更高。',
        focus: ['新闻与热点', '涨跌家数', '风险偏好变化'],
        route: '/chat',
        routeLabel: '追问 AI',
        icon: Radio,
      },
      {
        title: '宏观/市场复盘',
        status: 'partial',
        description: '承接大盘复盘、政策与经济环境观察，用于中长期仓位与行业判断。',
        focus: ['大盘阶段', '政策/利率/汇率', '资产配置语境'],
        route: '/',
        routeLabel: '运行大盘复盘',
        icon: Globe2,
      },
      {
        title: '量化/模型实验',
        status: 'available',
        description: 'Qlib、LightGBM、因子与回测结果在本地串联，MLflow 单独记录实验。',
        focus: ['因子与特征', '选股模型训练 / 交易策略回测', 'MLflow 实验记录'],
        route: '/backtest',
        routeLabel: '查看回测',
        icon: BrainCircuit,
      },
    ] satisfies CapabilityItem[],
    modes: [
      {
        title: '股票分析',
        status: 'available',
        icon: TrendingUp,
        description: '面向公司价值与交易信号，适合从个股代码进入完整 DSA 报告。',
        points: ['财报、估值、盈利和行业竞争力可作为核心输入', '技术、资金、新闻与量化信号共同形成观点', '入口复用首页、问股、告警、回测和选股页面'],
      },
      {
        title: '场内基金/ETF分析',
        status: 'partial',
        icon: PieChart,
        description: '作为一等研究对象展示，但不强行套用公司股票基本面框架。',
        points: ['支持趋势、均线、MACD、RSI/KDJ、成交量和流动性', '关注主题/指数逻辑、新闻与政策影响', '基本面部分降级：不按公司财报分析，不把 ROE、公司竞争力作为核心依据'],
      },
    ] satisfies ModeItem[],
    etfBoundaries: [
      {
        title: '可用',
        status: 'available',
        detail: '趋势、均线、MACD、RSI/KDJ、成交量、流动性、主题/指数逻辑、新闻/政策影响。',
      },
      {
        title: '部分可用',
        status: 'partial',
        detail: '基本面部分降级：不按公司财报分析；财报、ROE、公司竞争力不作为 ETF 核心分析依据。',
      },
      {
        title: '待补强',
        status: 'future',
        detail: '折溢价、跟踪误差、规模份额、持仓、基金类型、同类 ETF 对比。',
      },
    ] satisfies EtfBoundaryItem[],
    projectMap: [
      { name: 'daily_stock_analysis', role: '本地中文 WebUI 与综合分析入口', detail: '承接首页、问股、回测、告警、持仓和研究页。', icon: Network },
      { name: 'quant_workflow + qlib', role: '选股模型训练与实验记录', detail: '负责 Qlib 数据、因子、选股模型训练、LightGBM 选股模型基线和 MLflow。', icon: FlaskConical },
      { name: 'china-astock-quant', role: 'A股量化脚本与策略参考', detail: '提供选股、回测、规则策略和工程参考，可按需复用。', icon: Landmark },
      { name: 'quant_data', role: '本地数据仓库', detail: '沉淀 AkShare、JQData 和转换后的 Qlib 格式数据。', icon: Database },
    ] satisfies ProjectItem[],
    coverageItems: [
      { module: 'daily_stock_analysis WebUI', status: 'available', entry: '首页 / 研究 / 工作流 / 回测 / 问股 / 持仓 / 告警 / 选股', route: '/workflow', tests: '后端 pytest + 前端 vitest/lint/build', browser: 'local-workbench smoke 覆盖主页面', review: 'opencode复审通过', next: '继续把重任务体验做顺' },
      { module: 'quant_workflow + Qlib', status: 'available', entry: '工作流任务中心', route: '/workflow', tests: 'quant_workflow unittest + DSA quant API 测试', browser: '可从网页运行数据检查、质量报告、训练入口', review: 'opencode复审通过', next: '补更多产物预览和失败排查' },
      { module: 'china-astock-quant', status: 'available', entry: '工作流指标报告 / 研究页项目关系', route: '/workflow', tests: 'report_china_astock_indicators + run_china_astock_comparison 测试', browser: '可生成只读技术指标报告和策略对比报告', review: '待本阶段 opencode 复审', next: '继续评估事件驱动回测是否包装为只读对比报告' },
      { module: '场内基金/ETF', status: 'partial', entry: '研究页 ETF 模式', route: '/research', tests: '现有股票链路可复用，基金专用字段未补齐', browser: '边界提示已可见', review: '待 ETF 深度版复审', next: '补折溢价、跟踪误差、规模份额和持仓' },
      { module: '同花顺桌面端', status: 'future', entry: '后续外部工具', tests: '暂不读取本机窗口', browser: '未接入', review: '接入前先做只读方案评审', next: '只读观察，暂不接入交易' },
      { module: 'iPhone 镜像', status: 'future', entry: '暂缓', tests: '手机连接暂不做', browser: '未接入', review: '暂不审', next: '按当前要求保持延期' },
    ] satisfies SelfServiceCoverageItem[],
    developmentQueue: [
      { priority: 'P0', title: '每日研究向导', status: 'available', entry: '工作流 / 今日流程', route: '/workflow', next: '按网页顺序完成检查数据、质量报告、清洗预演、训练复盘和实验记录。', verification: 'Workflow vitest、local browser 切片、opencode复审均已覆盖。' },
      { priority: 'P1', title: '模型质量和清仓信号', status: 'partial', entry: '工作流 / 回测', route: '/workflow', next: '市场状态和现金仓位报告已可运行；下一步把 risk_off/defensive 信号接入组合回测和指标卡。', verification: '已有报告任务测试、浏览器预览和只读复审；后续仍需新增 Qlib 组合回测对比测试。' },
      { priority: 'P1', title: '场内基金/ETF深度版', status: 'partial', entry: '研究 / ETF模式', route: '/research', next: '补折溢价、跟踪误差、规模份额、持仓、基金类型和同类 ETF 对比。', verification: '先补数据字段和边界提示，再跑页面渲染与报告生成测试。' },
      { priority: 'P2', title: 'china-astock脚本白名单化', status: 'available', entry: '工作流任务中心', route: '/workflow', next: '指标库已接入工作流指标报告；下一步评估事件驱动回测是否包装为只读对比报告。', verification: 'report_china_astock_indicators 测试、后端白名单参数测试、报告预览浏览器测试、opencode安全复审。' },
      { priority: 'P2', title: '同花顺只读观察层', status: 'future', entry: '后续外部工具', next: '先做本机窗口/截图/OCR可行性方案，不读取密码、不自动下单。', verification: '实现前先做只读方案评审，确认没有交易执行路径。' },
    ] satisfies DevelopmentQueueItem[],
    futureItems: [
      { title: '本机炒股软件读取', status: '后续只读观察', description: '研究桌面行情软件的可读信息，不做登录凭据读取和交易执行。', icon: MonitorSmartphone },
      { title: 'iPhone 镜像 App 观察', status: '后续只读观察', description: '通过镜像界面观察手机 App 行情与组合信息，默认不读取密码、不自动下单。', icon: Smartphone },
      { title: '券商软件联动', status: '后续谨慎接入', description: '先做信息观察和人工确认流程，交易动作保持人工主导。', icon: ShieldAlert },
      { title: '聚宽/JQData 深度接入', status: '数据链路升级', description: '进一步补齐高质量历史数据、基本面数据和研究因子。', icon: BadgeCheck },
    ] satisfies FutureItem[],
  },
  en: {
    documentTitle: 'Research Workspace - DSA',
    eyebrow: 'Local research hub',
    title: 'Research Workspace',
    description: 'A local entry point that organizes stock analysis, ETF analysis, quant experiments, and future external-tool integrations.',
    openWorkflow: 'Open workflow',
    openMlflow: 'Open MLflow',
    backHome: 'Back home',
    statusTitle: 'Service Status',
    dataStatusTitle: 'Data Status',
    dataStatusDescription: 'Local data-quality and training status generated by quant_workflow; DSA only reads the artifact.',
    dataHygieneTitle: 'Data Hygiene',
    dataHygieneDescription: 'Local data inventory, cleanup plan, quality reports, and maintenance lock status.',
    rescanDataHygiene: 'Rescan',
    submittingDataHygiene: 'Submitting',
    scanningDataHygiene: 'Scanning',
    dataHygieneUpdated: 'Updated',
    dataHygieneFailed: 'Rescan failed',
    dataHygieneErrorSeparator: ': ',
    dataHygieneCancelled: 'Job cancelled',
    maintenanceLockAction: 'Maintenance lock active',
    maintenanceLockLabel: 'Maintenance Lock',
    locked: 'Locked',
    notLocked: 'Not locked',
    inventoryLabel: 'Inventory',
    cleanupPlanLabel: 'Cleanup Plan',
    qualityReportLabel: 'Quality Report',
    cleaningStrategyLabel: 'Cleaning Strategy',
    manifestLabel: 'Data Manifest',
    readmeLabel: 'Governance README',
    officialLabel: 'Official',
    candidateLabel: 'Candidate',
    legacyLabel: 'Legacy',
    archiveLabel: 'Archive',
    actionsLabel: 'Actions',
    dryRunLabel: 'Dry-run',
    viewReport: 'View Report',
    viewDetails: 'View Details',
    qualitySummaryTitle: 'Quality Summary',
    cleanupActionsTitle: 'Cleanup Actions (Preview)',
    actionTypeLabel: 'Type',
    sourceLabel: 'Source',
    destinationLabel: 'Destination',
    reasonLabel: 'Reason',
    riskLabel: 'Risk',
    confirmationRequired: 'Requires confirmation',
    stockUniverse: 'Stock Universe',
    usableInstruments: 'Usable Instruments',
    totalRows: 'Total Rows',
    dateRange: 'Date Range',
    trainable: 'Trainable',
    notTrainable: 'Not trainable yet',
    generatedAt: 'Generated',
    qlibProviderTitle: 'Qlib Provider',
    latestTrainingTitle: 'Latest Training Result',
    noTrainingResult: 'No training result yet',
    experimentNotebookTitle: 'Experiment Notebook',
    experimentNotebookDescription: 'Human-readable research notes for each quant run: intent, data version, key metrics, conclusion, and next step.',
    latestNoteTitle: 'Latest Human Note',
    latestRunsTitle: 'Latest Experiments',
    coverageTitle: 'Capability Acceptance Matrix',
    coverageDescription: 'Module-by-module status for browser self-service and the current verification gates: automated tests, browser execution, opencode review, and safety boundaries.',
    coverageModuleLabel: 'Module',
    coverageEntryLabel: 'Web entry',
    coverageTestsLabel: 'Automated tests',
    coverageBrowserLabel: 'Browser execution',
    coverageReviewLabel: 'opencode review',
    coverageNextLabel: 'Next',
    inventoryTitle: 'Function Entry Inventory',
    inventoryDescription: 'A ledger of existing DSA pages, quant workflow, Qlib integration, experiment records, and external observation. Core flows need a web entry or an explicit gap.',
    inventoryAreaLabel: 'Area',
    inventoryOwnerLabel: 'Owner',
    inventoryCoverageLabel: 'Current coverage',
    inventoryGapsLabel: 'Gaps',
    inventoryVerificationLabel: 'Verification',
    chinaReuseTitle: 'china-astock-quant Reuse List',
    chinaReuseDescription: 'Only research and backtest capabilities are candidates. Broker, trading, and automatic order modules remain reference-only and out of scope for this phase.',
    chinaReuseCapabilityLabel: 'Capability',
    chinaReuseSourceLabel: 'Source',
    chinaReuseDecisionLabel: 'Decision',
    chinaReuseTargetLabel: 'Target',
    chinaReuseRestrictionLabel: 'Restrictions',
    acceptanceTitle: 'Acceptance Checklist',
    acceptanceDescription: 'A page-level checklist for whether this phase is actually shippable, not just verbally described.',
    acceptanceEvidenceLabel: 'Evidence',
    acceptanceNextLabel: 'Next',
    developmentQueueTitle: 'Development Queue',
    developmentQueueDescription: 'Next work is split into verifiable modules. Each item must pass automated tests, browser execution, and read-only review.',
    developmentPriorityLabel: 'Priority',
    developmentEntryLabel: 'Web entry',
    developmentActionLabel: 'Next action',
    developmentVerificationLabel: 'Verification',
    experimentComparisonTitle: 'Cleaned vs Canonical',
    experimentComparisonDescription: 'Core backtest metric for the same baseline on native V2 data versus legacy cleaned data.',
    cleanedDataLabel: 'Cleaned data',
    canonicalDataLabel: 'Native V2 data',
    comparisonMetricLabel: 'Annualized excess return with cost',
    comparisonDeltaLabel: 'Delta',
    comparisonNoData: 'At least one cleaned run and one canonical run are required for comparison.',
    noExperimentNote: 'No human-readable experiment note yet. Run training first, then write conclusions to experiment_log.md.',
    noExperimentRuns: 'No structured experiment entries yet.',
    archiveExperiment: 'Archive',
    archiveExperimentAria: 'Archive experiment',
    permanentDeleteExperiment: 'Delete permanently',
    permanentDeleteExperimentAria: 'Permanently delete experiment',
    permanentDeleteConfirm: 'Permanently delete this experiment record? This removes it from experiment_runs.jsonl and removes the marked run_id block from experiment_log.md. MLflow and training files are not deleted.',
    deleteExperimentFailed: 'Experiment record operation failed',
    runIdLabel: 'Run',
    modelLabel: 'Model',
    strategyLabel: 'Strategy',
    datasetLabel: 'Dataset range',
    conclusionLabel: 'Conclusion',
    nextStepLabel: 'Next step',
    notebookFileLabel: 'Notebook file',
    statusLoadFailed: 'Status load failed',
    filesLabel: 'Files',
    rowsLabel: 'Rows',
    instrumentsLabel: 'Instruments',
    calendarLabel: 'Calendar',
    capabilitiesTitle: 'Six Analysis Capabilities',
    capabilitiesDescription: 'The first version is an entry map and capability boundary; analysis still reuses the existing DSA, Qlib, and data workflow.',
    dualModeTitle: 'Stock And ETF Modes',
    dualModeDescription: 'The same workspace separates different research objects, so ETFs are not treated like ordinary operating companies.',
    etfBoundaryTitle: 'ETF First-version Boundary',
    projectMapTitle: 'Project Map',
    projectMapDescription: 'Each project keeps its role; DSA is the local research entry and result aggregation layer.',
    futureTitle: 'Future Upgrades',
    futureDescription: 'These items are roadmap only in this round: read-only observation by default, no password capture, no automatic orders.',
    externalToolsTitle: 'External Research Tools',
    externalToolsDescription: 'Use jump links first, then later decide whether data and modules should be merged.',
    catdeskTitle: 'CatDesk 9 / US catalysts to A-share morning report',
    catdeskDescription: 'Cross-market catalyst observation: rank A-share candidates from overnight US sector and theme moves. Its backtest evidence is candidate-signal evidence, not a Qlib strategy backtest.',
    catdeskOpen: 'Open CatDesk 9',
    catdeskNote: 'Current integration is link-only; data is not shared yet. Module de-duplication can be reviewed later.',
    mlflowNote: 'MLflow remains the experiment-tracking page. It is not embedded or modified; use the workflow header for live status.',
    portNote: 'DSA follows the current browser address; local port 8000 may belong to another service.',
    statusItems: [
      { label: 'DSA service', value: 'Current page', detail: 'Local WebUI', tone: 'success' },
      { label: 'Data sources', value: 'AkShare/JQData/local data', detail: 'Existing workflow', tone: 'info' },
      { label: 'MLflow', value: 'Experiment tracking', detail: '/experiments/ (status on Workflow)', tone: 'info' },
      { label: 'Port', value: 'Current browser address', detail: 'No hard-coded 8000', tone: 'warning' },
    ] satisfies StatusItem[],
    capabilities: [
      {
        title: 'Fundamental analysis',
        status: 'partial',
        description: 'Stocks can use financials, earnings, valuation, industry, and moat analysis; ETF fundamentals downgrade to index and theme logic.',
        focus: ['Financials / earnings / valuation', 'Industry and competitive strength', 'ETF theme and index downgrade'],
        route: '/',
        routeLabel: 'Analyze on Home',
        icon: Building2,
      },
      {
        title: 'Technical analysis',
        status: 'available',
        description: 'Reuse price series, moving averages, MACD, RSI/KDJ, volume, and pattern context.',
        focus: ['Candles and moving averages', 'MACD / RSI / KDJ', 'Price-volume structure'],
        route: '/alerts',
        routeLabel: 'Configure alerts',
        icon: LineChart,
      },
      {
        title: 'Capital-flow analysis',
        status: 'partial',
        description: 'Stocks can use money flow, margin data, and trading lists; ETFs emphasize turnover, liquidity, and traded value.',
        focus: ['Major / northbound flow', 'Margin and active trading lists', 'ETF value traded and liquidity'],
        route: '/',
        routeLabel: 'View analysis',
        icon: WalletCards,
      },
      {
        title: 'Sentiment analysis',
        status: 'partial',
        description: 'Uses market psychology, themes, news, and breadth as supporting evidence, with higher weight for short-term timing.',
        focus: ['News and themes', 'Market breadth', 'Risk appetite changes'],
        route: '/chat',
        routeLabel: 'Ask AI',
        icon: Radio,
      },
      {
        title: 'Macro / market review',
        status: 'partial',
        description: 'Connects market reviews, policy, and economic context for medium-term allocation and sector judgment.',
        focus: ['Market phase', 'Policy / rates / FX', 'Allocation context'],
        route: '/',
        routeLabel: 'Run market review',
        icon: Globe2,
      },
      {
        title: 'Quant / model experiments',
        status: 'available',
        description: 'Qlib, LightGBM, factors, and backtests are connected locally; MLflow records experiments separately.',
        focus: ['Factors and features', 'Training and backtests', 'MLflow experiment tracking'],
        route: '/backtest',
        routeLabel: 'View backtests',
        icon: BrainCircuit,
      },
    ] satisfies CapabilityItem[],
    modes: [
      {
        title: 'Stock analysis',
        status: 'available',
        icon: TrendingUp,
        description: 'Company-value and trading-signal analysis, best entered through a stock code and a full DSA report.',
        points: ['Financials, valuation, earnings, and industry strength are core inputs', 'Technical, flow, news, and quant signals form the final view', 'Reuses Home, Ask, Alerts, Backtest, and Screening pages'],
      },
      {
        title: 'Exchange-traded fund / ETF analysis',
        status: 'partial',
        icon: PieChart,
        description: 'A first-class research object, but not forced into a company-stock fundamental framework.',
        points: ['Supports trend, moving averages, MACD, RSI/KDJ, volume, and liquidity', 'Focuses on theme/index logic plus news and policy impact', 'ETF company financial statements are downgraded: do not analyze it like company financial statements, ROE, or corporate competitiveness'],
      },
    ] satisfies ModeItem[],
    etfBoundaries: [
      {
        title: 'Available',
        status: 'available',
        detail: 'Trend, moving averages, MACD, RSI/KDJ, volume, liquidity, theme/index logic, news and policy impact.',
      },
      {
        title: 'Partial',
        status: 'partial',
        detail: 'ETF company financial statements are downgraded: financial reports, ROE, and corporate competitiveness are not core ETF evidence.',
      },
      {
        title: 'To strengthen',
        status: 'future',
        detail: 'Premium/discount, tracking error, AUM/shares, holdings, fund type, and peer ETF comparison.',
      },
    ] satisfies EtfBoundaryItem[],
    projectMap: [
      { name: 'daily_stock_analysis', role: 'Local Chinese WebUI and combined analysis entry', detail: 'Hosts Home, Ask, Backtest, Alerts, Portfolio, and Research pages.', icon: Network },
      { name: 'quant_workflow + qlib', role: 'Quant training and experiment tracking', detail: 'Owns Qlib data, factors, model training, LightGBM baseline, and MLflow.', icon: FlaskConical },
      { name: 'china-astock-quant', role: 'A-share quant scripts and strategy reference', detail: 'Offers screening, backtest, rule strategy, and engineering references for reuse.', icon: Landmark },
      { name: 'quant_data', role: 'Local data warehouse', detail: 'Stores AkShare, JQData, and converted Qlib-format datasets.', icon: Database },
    ] satisfies ProjectItem[],
    coverageItems: [
      { module: 'daily_stock_analysis WebUI', status: 'available', entry: 'Home / Research / Workflow / Backtest / Ask / Portfolio / Alerts / Screening', route: '/workflow', tests: 'Backend pytest + frontend vitest/lint/build', browser: 'local-workbench smoke covers main pages', review: 'opencode review passed', next: 'Improve heavy-task ergonomics' },
      { module: 'quant_workflow + Qlib', status: 'available', entry: 'Workflow job center', route: '/workflow', tests: 'quant_workflow unittest + DSA quant API tests', browser: 'Data check, quality reports, and training entry are runnable from the browser', review: 'opencode review passed', next: 'Add richer artifact previews and failure triage' },
      { module: 'china-astock-quant', status: 'available', entry: 'Workflow indicator report / Research project map', route: '/workflow', tests: 'report_china_astock_indicators + run_china_astock_comparison tests', browser: 'Can generate read-only technical indicator and strategy comparison reports', review: 'Pending this-stage opencode review', next: 'Evaluate whether event-driven backtests should be wrapped as read-only comparison reports' },
      { module: 'Exchange-traded funds / ETF', status: 'partial', entry: 'Research ETF mode', route: '/research', tests: 'Existing stock workflow can be reused; fund-specific fields are missing', browser: 'Boundary notice is visible', review: 'Review after ETF deep version', next: 'Add premium/discount, tracking error, AUM/shares, and holdings' },
      { module: 'Tonghuashun desktop', status: 'future', entry: 'Future external tools', tests: 'No local window reading yet', browser: 'Not integrated', review: 'Read-only design review before integration', next: 'Read-only observation; no trading integration yet' },
      { module: 'iPhone mirroring', status: 'future', entry: 'Deferred', tests: 'Phone connection deferred', browser: 'Not integrated', review: 'Not reviewed yet', next: 'Keep deferred per current scope' },
    ] satisfies SelfServiceCoverageItem[],
    developmentQueue: [
      { priority: 'P0', title: 'Daily research guide', status: 'available', entry: 'Workflow / daily flow', route: '/workflow', next: 'Use the browser flow for data checks, quality reports, cleaning dry-run, training review, and experiment notes.', verification: 'Workflow vitest, local browser slice, and opencode review are covered.' },
      { priority: 'P1', title: 'Model quality and cash signal', status: 'partial', entry: 'Workflow / Backtest', route: '/workflow', next: 'Market-regime and cash-ratio report is runnable; next wire risk_off/defensive signals into portfolio backtests and metric cards.', verification: 'Report job tests, browser preview, and read-only review are covered; Qlib portfolio comparison tests are still needed.' },
      { priority: 'P1', title: 'ETF deep version', status: 'partial', entry: 'Research / ETF mode', route: '/research', next: 'Add premium/discount, tracking error, AUM/shares, holdings, fund type, and peer ETF comparison.', verification: 'Add data fields and boundary notices first, then page-render and report-generation tests.' },
      { priority: 'P2', title: 'china-astock script whitelist', status: 'available', entry: 'Workflow job center', route: '/workflow', next: 'Indicator functions are wired into the workflow indicator report; next evaluate event-driven backtests as read-only comparison reports.', verification: 'report_china_astock_indicators tests, backend whitelist tests, browser artifact preview, and opencode security review.' },
      { priority: 'P2', title: 'Tonghuashun read-only observation', status: 'future', entry: 'Future external tools', next: 'Design local window/screenshot/OCR feasibility first; no password reading and no automatic orders.', verification: 'Read-only design review before implementation; confirm no trading execution path.' },
    ] satisfies DevelopmentQueueItem[],
    futureItems: [
      { title: 'Desktop trading software reading', status: 'Future read-only observation', description: 'Study readable market-software information without reading login credentials or executing trades.', icon: MonitorSmartphone },
      { title: 'iPhone mirrored app observation', status: 'Future read-only observation', description: 'Observe mobile-app quotes and portfolio screens through mirroring; no passwords and no automatic orders by default.', icon: Smartphone },
      { title: 'Broker software linkage', status: 'Future cautious integration', description: 'Start with observation and human confirmation; trading actions remain human-led.', icon: ShieldAlert },
      { title: 'JoinQuant/JQData deep integration', status: 'Data workflow upgrade', description: 'Improve high-quality historical data, fundamental data, and research factors.', icon: BadgeCheck },
    ] satisfies FutureItem[],
  },
} as const;

function statusLabel(status: SupportState, language: 'zh' | 'en'): string {
  return STATUS_STYLE[status][language];
}

const StatusBadge: React.FC<{ status: SupportState; language: 'zh' | 'en' }> = ({ status, language }) => {
  const config = STATUS_STYLE[status];
  return (
    <Badge variant={config.variant} size="sm" className="shrink-0">
      <StatusDot tone={config.tone} className="h-1.5 w-1.5" />
      {statusLabel(status, language)}
    </Badge>
  );
};

const EntryLink: React.FC<{ to: string; children: React.ReactNode }> = ({ to, children }) => (
  <Link
    to={to}
    className="inline-flex h-9 items-center justify-center gap-2 rounded-lg border border-cyan/25 px-3 text-sm font-medium text-cyan transition-colors hover:bg-cyan/10"
  >
    {children}
    <ArrowRight className="h-4 w-4" />
  </Link>
);

function sourceStatusTone(status: ResearchDataSource['status']): { variant: BadgeVariant; tone: StatusTone } {
  const meta = DATA_STATUS_STYLE[status] ?? DATA_STATUS_STYLE.unknown;
  return { variant: meta.variant, tone: meta.tone };
}

function dataStatusLabel(status: string | undefined, language: 'zh' | 'en'): string {
  return (DATA_STATUS_STYLE[status ?? 'missing'] ?? DATA_STATUS_STYLE.unknown)[language];
}

function dataStatusMeta(status: string | undefined): { variant: BadgeVariant; tone: StatusTone } {
  const meta = DATA_STATUS_STYLE[status ?? 'missing'] ?? DATA_STATUS_STYLE.unknown;
  return { variant: meta.variant, tone: meta.tone };
}

function formatCompactNumber(value: number | undefined): string {
  if (value === undefined || Number.isNaN(value)) {
    return '-';
  }
  return new Intl.NumberFormat('en-US').format(value);
}

function formatMetricValue(value: number): string {
  if (!Number.isFinite(value)) {
    return '-';
  }
  return Math.abs(value) < 1 ? value.toFixed(4) : value.toFixed(2);
}

function formatSignedMetricValue(value: number): string {
  const formatted = formatMetricValue(value);
  if (formatted === '-') {
    return formatted;
  }
  return value > 0 ? `+${formatted}` : formatted;
}

function marketMetricClass(value: number): string {
  if (value > 0) {
    return 'text-market-up';
  }
  if (value < 0) {
    return 'text-market-down';
  }
  return 'text-foreground';
}

function runText(run: ResearchExperimentRun, field: 'provider' | 'experimentName' | 'qlibProvider' | 'providerUri'): string {
  return String(run[field] ?? '').toLowerCase();
}

function isCleanedExperiment(run: ResearchExperimentRun): boolean {
  return (
    runText(run, 'provider') === 'cleaned'
    || runText(run, 'experimentName').includes('cleaned')
    || runText(run, 'qlibProvider').includes('_cleaned')
    || runText(run, 'providerUri').includes('_cleaned')
  );
}

function isCanonicalExperiment(run: ResearchExperimentRun, cleanedRun: ResearchExperimentRun | undefined): boolean {
  if (run === cleanedRun) {
    return false;
  }
  const providerPath = runText(run, 'qlibProvider') || runText(run, 'providerUri');
  return providerPath.includes('cn_data_ashare_v2') && !providerPath.includes('_cleaned');
}

function metricValue(run: ResearchExperimentRun | undefined, metricKey: string): number | undefined {
  const value = run?.metrics?.[metricKey];
  return value !== undefined && Number.isFinite(value) ? value : undefined;
}

function buildExperimentComparison(runs: ResearchExperimentRun[]): ExperimentComparison | null {
  const cleanedRun = runs.find(isCleanedExperiment);
  const canonicalRun = runs.find((run) => isCanonicalExperiment(run, cleanedRun));
  const cleanedValue = metricValue(cleanedRun, COMPARISON_METRIC_KEY);
  const canonicalValue = metricValue(canonicalRun, COMPARISON_METRIC_KEY);

  if (!cleanedRun || !canonicalRun || cleanedValue === undefined || canonicalValue === undefined) {
    return null;
  }

  return {
    cleanedRun,
    canonicalRun,
    metricKey: COMPARISON_METRIC_KEY,
    cleanedValue,
    canonicalValue,
    delta: cleanedValue - canonicalValue,
  };
}

function joinResearchList(items: string[], language: 'zh' | 'en'): string {
  if (!items.length) {
    return '-';
  }
  return items.join(language === 'zh' ? '、' : ', ');
}

function cleanupActionSourceLabel(source: string | undefined, language: 'zh' | 'en'): string {
  const clean = (source || '').trim();
  if (!clean) {
    return language === 'zh' ? '未指定来源' : 'Source not specified';
  }
  return clean.split('/').filter(Boolean).pop() || clean;
}

function externalToolStatus(status: string | undefined, installed?: boolean): SupportState {
  if (status === 'available' || installed) {
    return 'available';
  }
  if (status === 'partial') {
    return 'partial';
  }
  return 'future';
}

function normalizeSupportState(status: string | undefined): SupportState {
  if (
    status === 'available'
    || status === 'partial'
    || status === 'future'
    || status === 'read_only'
    || status === 'deferred'
    || status === 'not_planned'
  ) {
    return status;
  }
  if (status === 'verified') {
    return 'available';
  }
  if (status === 'missing') {
    return 'deferred';
  }
  return 'future';
}

function buildCapabilityMatrixItems(
  matrix: ResearchCapabilityMatrixItem[] | null | undefined,
  language: 'zh' | 'en',
): SelfServiceCoverageItem[] | null {
  if (!matrix?.length) {
    return null;
  }
  return matrix.map((item) => {
    const restrictions = item.restrictions ?? [];
    const restrictionText = restrictions.length
      ? `${language === 'zh' ? '限制' : 'Restrictions'}: ${joinResearchList(restrictions, language)}`
      : '';
    return {
      module: item.module,
      status: normalizeSupportState(item.status),
      entry: item.entry,
      route: item.route,
      tests: item.tests,
      browser: item.browser,
      review: item.review,
      next: restrictionText ? `${item.next} ${restrictionText}` : item.next,
      restrictions,
    };
  });
}

function buildCoverageItems(
  items: readonly SelfServiceCoverageItem[],
  externalTools: ResearchExternalTools | null | undefined,
  language: 'zh' | 'en',
): SelfServiceCoverageItem[] {
  const tonghuashun = externalTools?.tonghuashun;
  const iphoneMirroring = externalTools?.iphoneMirroring;
  return items.map((item) => {
    const isTonghuashun = item.module === '同花顺桌面端' || item.module === 'Tonghuashun desktop';
    if (isTonghuashun && tonghuashun) {
      return {
        ...item,
        status: externalToolStatus(tonghuashun.status, tonghuashun.installed),
        entry: language === 'zh' ? '研究页 / 本机只读检测' : 'Research / local read-only detection',
        route: '/research',
        tests: language === 'zh'
          ? 'research_status API + ResearchPage 测试'
          : 'research_status API + ResearchPage tests',
        browser: `${tonghuashun.summary} ${language === 'zh' ? '能力' : 'Capabilities'}: ${joinResearchList(tonghuashun.capabilities, language)}`,
        review: language === 'zh' ? 'opencode复审通过' : 'opencode review passed',
        next: joinResearchList(tonghuashun.restrictions, language),
      };
    }

    const isIphone = item.module === 'iPhone 镜像' || item.module === 'iPhone mirroring';
    if (isIphone && iphoneMirroring) {
      return {
        ...item,
        status: 'future',
        browser: iphoneMirroring.summary,
        next: joinResearchList(iphoneMirroring.restrictions, language),
      };
    }
    return item;
  });
}

const ResearchPage: React.FC = () => {
  const { language } = useUiLanguage();
  const text = RESEARCH_TEXT[language];
  const [researchStatus, setResearchStatus] = useState<ResearchStatus | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [mutatingRunKey, setMutatingRunKey] = useState<string | null>(null);
  const [confirmDeleteRunId, setConfirmDeleteRunId] = useState<string | null>(null);
  const [dataHygieneJob, setDataHygieneJob] = useState<QuantJob | null>(null);
  const [dataHygieneSubmitting, setDataHygieneSubmitting] = useState(false);
  const [dataHygieneJobError, setDataHygieneJobError] = useState<string | null>(null);
  const [dataHygieneUpdated, setDataHygieneUpdated] = useState(false);
  const dataHygieneBusyRef = useRef(false);
  const dataHygienePollTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    document.title = text.documentTitle;
  }, [text.documentTitle]);

  const loadResearchStatus = useCallback(async () => {
    try {
      const status = await researchApi.getStatus();
      if (!mountedRef.current) {
        return false;
      }
      setResearchStatus(status);
      setStatusError(null);
      return true;
    } catch (error: unknown) {
      if (!mountedRef.current) {
        return false;
      }
      const message = error instanceof Error ? error.message : String(error);
      setStatusError(message);
      return false;
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    void loadResearchStatus();

    return () => {
      mountedRef.current = false;
      if (dataHygienePollTimerRef.current !== null) {
        clearTimeout(dataHygienePollTimerRef.current);
      }
    };
  }, [loadResearchStatus]);

  const pollDataHygieneJob = useCallback(async (jobId: string): Promise<void> => {
    try {
      const job = await quantApi.getJob(jobId);
      if (!mountedRef.current) {
        return;
      }
      setDataHygieneJob(job);

      if (job.status === 'completed') {
        dataHygieneBusyRef.current = false;
        const refreshed = await loadResearchStatus();
        if (mountedRef.current && refreshed) {
          setDataHygieneUpdated(true);
        }
        return;
      }

      if (job.status === 'failed' || job.status === 'cancelled') {
        dataHygieneBusyRef.current = false;
        setDataHygieneJobError(job.error || (job.status === 'cancelled' ? text.dataHygieneCancelled : job.status));
        return;
      }

      dataHygienePollTimerRef.current = setTimeout(() => {
        void pollDataHygieneJob(jobId);
      }, 1000);
    } catch (error: unknown) {
      if (!mountedRef.current) {
        return;
      }
      dataHygieneBusyRef.current = false;
      const message = error instanceof Error ? error.message : String(error);
      setDataHygieneJobError(message);
    }
  }, [loadResearchStatus, text.dataHygieneCancelled]);

  const handleDataHygieneRescan = async () => {
    if (researchStatus?.dataHygiene?.maintenanceLock.locked || dataHygieneBusyRef.current) {
      return;
    }
    dataHygieneBusyRef.current = true;
    setDataHygieneSubmitting(true);
    setDataHygieneJobError(null);
    setDataHygieneUpdated(false);
    try {
      const job = await quantApi.submitJob('report_data_hygiene', {});
      if (!mountedRef.current) {
        return;
      }
      setDataHygieneJob(job);
      setDataHygieneSubmitting(false);
      await pollDataHygieneJob(job.jobId);
    } catch (error: unknown) {
      if (!mountedRef.current) {
        return;
      }
      dataHygieneBusyRef.current = false;
      setDataHygieneSubmitting(false);
      const message = error instanceof Error ? error.message : String(error);
      setDataHygieneJobError(message);
    }
  };

  const trainableLabel = researchStatus?.trainable ? text.trainable : text.notTrainable;
  const trainableTone: StatusTone = researchStatus?.trainable ? 'success' : 'warning';
  const experimentNotebook = researchStatus?.experimentNotebook;
  const experimentRuns = experimentNotebook?.runs ?? [];
  const experimentComparison = buildExperimentComparison(experimentRuns);
  const experimentRunCount = experimentNotebook?.totalRuns ?? experimentRuns.length;
  const mlflowUrl = resolveMlflowUiUrl(researchStatus?.mlflow?.url || DEFAULT_MLFLOW_URL);
  const mlflowStatus = researchStatus?.mlflow?.status || 'linked';
  const mlflowStatusMeta = dataStatusMeta(mlflowStatus);
  const mlflowStatusLabel = dataStatusLabel(mlflowStatus, language);
  const statusItems = text.statusItems.map((item) => (
    item.label === 'MLflow'
      ? {
          ...item,
          value: mlflowStatusLabel,
          detail: mlflowUrl,
          tone: mlflowStatusMeta.tone,
        }
      : item
  ));
  const experimentCountLabel = language === 'zh'
    ? `最近 ${experimentRunCount} 次实验`
    : `Latest ${experimentRunCount} experiment${experimentRunCount === 1 ? '' : 's'}`;
  const notebookStatus = experimentNotebook?.status ?? 'missing';
  const notebookStatusMeta = DATA_STATUS_STYLE[notebookStatus] ?? DATA_STATUS_STYLE.unknown;
  const notebookStatusVariant: BadgeVariant = notebookStatusMeta.variant;
  const notebookStatusTone: StatusTone = notebookStatusMeta.tone;
  const coverageItems = buildCapabilityMatrixItems(researchStatus?.capabilityMatrix, language)
    ?? buildCoverageItems(text.coverageItems, researchStatus?.externalTools, language);
  const functionInventoryItems: FunctionInventoryItem[] = researchStatus?.functionInventory ?? [];
  const chinaReuseItems: ChinaAstockReuseItem[] = researchStatus?.chinaAstockReuse ?? [];
  const acceptanceItems: AcceptanceChecklistItem[] = researchStatus?.acceptanceChecklist ?? [];
  const dataHygieneLocked = Boolean(researchStatus?.dataHygiene?.maintenanceLock.locked);
  const dataHygieneRunning = dataHygieneSubmitting
    || dataHygieneJob?.status === 'pending'
    || dataHygieneJob?.status === 'running';
  const dataHygieneButtonLabel = dataHygieneLocked
    ? text.maintenanceLockAction
    : dataHygieneSubmitting
      ? text.submittingDataHygiene
      : dataHygieneRunning
        ? text.scanningDataHygiene
        : text.rescanDataHygiene;

  const applyExperimentMutation = async (runId: string, action: 'archive' | 'permanent') => {
    setMutatingRunKey(`${action}:${runId}`);
    setDeleteError(null);
    try {
      const result = action === 'archive'
        ? await researchApi.archiveExperimentRun(runId)
        : await researchApi.permanentlyDeleteExperimentRun(runId);
      setResearchStatus((current) => {
        if (!current) {
          return current;
        }
        return {
          ...current,
          experimentNotebook: result.experimentNotebook,
        };
      });
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : String(error);
      setDeleteError(message);
    } finally {
      setMutatingRunKey(null);
    }
  };

  const handleArchiveExperiment = async (runId: string) => {
    await applyExperimentMutation(runId, 'archive');
  };

  const handlePermanentDeleteExperiment = (runId: string) => {
    setConfirmDeleteRunId(runId);
  };

  const handleConfirmPermanentDelete = async () => {
    const runId = confirmDeleteRunId;
    setConfirmDeleteRunId(null);
    if (runId) {
      await applyExperimentMutation(runId, 'permanent');
    }
  };

  const handleCancelPermanentDelete = () => {
    setConfirmDeleteRunId(null);
  };

  return (
    <AppPage className="max-w-[1500px] space-y-5">
      <PageHeader
        eyebrow={text.eyebrow}
        title={text.title}
        description={text.description}
        actions={(
          <>
            <Link
              to="/workflow"
              className="inline-flex h-10 items-center gap-2 rounded-xl border border-cyan/30 bg-cyan/10 px-4 text-sm font-medium text-cyan transition-colors hover:bg-cyan/15"
            >
              <ArrowRight className="h-4 w-4" />
              {text.openWorkflow}
            </Link>
            <a
              href={mlflowUrl}
              target="_blank"
              rel="noreferrer"
              className="inline-flex h-10 items-center gap-2 rounded-xl border border-cyan/30 bg-cyan/10 px-4 text-sm font-medium text-cyan transition-colors hover:bg-cyan/15"
            >
              <ArrowUpRight className="h-4 w-4" />
              {text.openMlflow}
            </a>
            <Link
              to="/"
              className="inline-flex h-10 items-center gap-2 rounded-xl border border-border/70 bg-card px-4 text-sm font-medium text-foreground shadow-soft-card transition-colors hover:bg-hover"
            >
              <Search className="h-4 w-4" />
              {text.backHome}
            </Link>
          </>
        )}
      />

      <section aria-label={text.statusTitle} className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        {statusItems.map((item) => (
          <Card key={item.label} padding="sm" className="min-h-[112px]">
            <div className="flex h-full items-start justify-between gap-3">
              <div className="min-w-0">
                <p className="text-xs font-medium text-muted-text">{item.label}</p>
                <p className="mt-2 truncate text-sm font-semibold text-foreground">{item.value}</p>
                <p className="mt-1 text-xs leading-5 text-secondary-text">{item.detail}</p>
              </div>
              <StatusDot tone={item.tone} pulse={item.tone === 'success'} className="mt-1" />
            </div>
          </Card>
        ))}
      </section>

      <section aria-label={text.dataStatusTitle}>
        <Card padding="lg" className="space-y-5">
          <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
            <div>
              <h2 className="text-lg font-semibold text-foreground">{text.dataStatusTitle}</h2>
              <p className="mt-1 max-w-3xl text-sm leading-6 text-secondary-text">{text.dataStatusDescription}</p>
              {researchStatus?.generatedAt ? (
                <p className="mt-2 text-xs text-muted-text">
                  {text.generatedAt}: {researchStatus.generatedAt}
                </p>
              ) : null}
              {statusError ? (
                <p className="mt-2 text-xs text-danger">
                  {text.statusLoadFailed}: {statusError}
                </p>
              ) : null}
            </div>
            <Badge variant={researchStatus?.trainable ? 'success' : 'warning'} size="sm" className="w-fit">
              <StatusDot tone={trainableTone} className="h-1.5 w-1.5" />
              {trainableLabel}
            </Badge>
          </div>

          {researchStatus?.trainingReadiness?.summary ? (
            <div className="rounded-xl border border-border/70 bg-surface/70 px-4 py-3 text-sm leading-6 text-secondary-text">
              {researchStatus.trainingReadiness.summary}
            </div>
          ) : null}

          <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_300px_300px]">
            <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
              {(researchStatus?.dataSources ?? []).map((source) => {
                const sourceTone = sourceStatusTone(source.status);
                return (
                  <div key={source.name} className="rounded-xl border border-border/70 bg-surface/60 p-3">
                    <div className="mb-3 flex items-center justify-between gap-2">
                      <h3 className="break-words text-sm font-semibold text-foreground">{source.name}</h3>
                      <Badge variant={sourceTone.variant} size="sm">
                        <StatusDot tone={sourceTone.tone} className="h-1.5 w-1.5" />
                        {dataStatusLabel(source.status, language)}
                      </Badge>
                    </div>
                    <dl className="space-y-1 text-xs text-secondary-text">
                      <div className="flex justify-between gap-3">
                        <dt>{text.filesLabel}</dt>
                        <dd className="font-medium text-foreground">{formatCompactNumber(source.files)}</dd>
                      </div>
                      <div className="flex justify-between gap-3">
                        <dt>{text.rowsLabel}</dt>
                        <dd className="font-medium text-foreground">{formatCompactNumber(source.rows)}</dd>
                      </div>
                      <div className="pt-1 leading-5">{source.dateRange}</div>
                    </dl>
                  </div>
                );
              })}
            </div>

            <div className="rounded-xl border border-border/70 bg-surface/60 p-3">
              <h3 className="text-sm font-semibold text-foreground">{text.qlibProviderTitle}</h3>
              <dl className="mt-3 space-y-2 text-xs text-secondary-text">
                <div className="flex justify-between gap-3">
                  <dt>{text.instrumentsLabel}</dt>
                  <dd className="font-medium text-foreground">{formatCompactNumber(researchStatus?.qlibProvider?.instruments)}</dd>
                </div>
                <div className="flex justify-between gap-3">
                  <dt>{text.calendarLabel}</dt>
                  <dd className="font-medium text-foreground">{formatCompactNumber(researchStatus?.qlibProvider?.calendarDays)}</dd>
                </div>
                <div className="leading-5">{researchStatus?.qlibProvider?.calendarRange ?? '-'}</div>
              </dl>
            </div>

            <div className="rounded-xl border border-border/70 bg-surface/60 p-3">
              <h3 className="text-sm font-semibold text-foreground">{text.latestTrainingTitle}</h3>
              {researchStatus?.latestTraining ? (
                <div className="mt-3 space-y-2 text-xs text-secondary-text">
                  <p className="font-medium text-foreground">
                    {researchStatus.latestTraining.experimentName ?? researchStatus.latestTraining.status}
                  </p>
                  {researchStatus.latestTraining.runId ? (
                    <p className="break-all font-mono">{researchStatus.latestTraining.runId}</p>
                  ) : null}
                  <div className="flex flex-wrap gap-2 pt-1">
                    {Object.entries(researchStatus.latestTraining.metrics ?? {}).slice(0, 4).map(([name, value]) => (
                      <Badge key={name} variant="history" size="sm" className="rounded-lg">
                        {name}: {formatMetricValue(value)}
                      </Badge>
                    ))}
                  </div>
                </div>
              ) : (
                <p className="mt-3 text-xs leading-5 text-secondary-text">{text.noTrainingResult}</p>
              )}
            </div>
          </div>
        </Card>
      </section>

      {researchStatus?.dataHygiene ? (
        <section aria-label={text.dataHygieneTitle}>
          <Card padding="lg" className="space-y-4">
            <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
              <div>
                <h2 className="text-lg font-semibold text-foreground">{text.dataHygieneTitle}</h2>
                <p className="mt-1 max-w-3xl text-sm leading-6 text-secondary-text">{text.dataHygieneDescription}</p>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <Badge
                  variant={researchStatus.dataHygiene.maintenanceLock.locked ? 'warning' : 'success'}
                  size="sm"
                  className="w-fit"
                >
                  <StatusDot
                    tone={researchStatus.dataHygiene.maintenanceLock.locked ? 'warning' : 'success'}
                    className="h-1.5 w-1.5"
                  />
                  {text.maintenanceLockLabel}: {researchStatus.dataHygiene.maintenanceLock.locked ? text.locked : text.notLocked}
                </Badge>
                <Button
                  variant="secondary"
                  size="sm"
                  disabled={dataHygieneLocked || dataHygieneRunning}
                  isLoading={dataHygieneRunning}
                  loadingText={dataHygieneButtonLabel}
                  onClick={() => void handleDataHygieneRescan()}
                >
                  <RefreshCw className="h-4 w-4" aria-hidden="true" />
                  {dataHygieneButtonLabel}
                </Button>
                {dataHygieneUpdated ? (
                  <span className="text-xs font-medium text-success" role="status">{text.dataHygieneUpdated}</span>
                ) : null}
              </div>
            </div>

            {dataHygieneJobError ? (
              <p className="text-sm text-danger" role="alert">
                {text.dataHygieneFailed}{text.dataHygieneErrorSeparator}{dataHygieneJobError}
              </p>
            ) : null}

            <div className="flex flex-wrap gap-2">
              {researchStatus.dataHygiene.manifest ? (
                <Badge
                  variant={researchStatus.dataHygiene.manifest.status === 'available' ? 'success' : 'default'}
                  size="sm"
                  className="gap-1.5 shadow-none"
                >
                  <StatusDot
                    tone={researchStatus.dataHygiene.manifest.status === 'available' ? 'success' : 'neutral'}
                    className="h-1.5 w-1.5"
                  />
                  {text.manifestLabel}: {dataStatusLabel(researchStatus.dataHygiene.manifest.status, language)}
                </Badge>
              ) : null}
              {researchStatus.dataHygiene.readme ? (
                <Badge
                  variant={researchStatus.dataHygiene.readme.status === 'available' ? 'success' : 'default'}
                  size="sm"
                  className="gap-1.5 shadow-none"
                >
                  <StatusDot
                    tone={researchStatus.dataHygiene.readme.status === 'available' ? 'success' : 'neutral'}
                    className="h-1.5 w-1.5"
                  />
                  {text.readmeLabel}: {dataStatusLabel(researchStatus.dataHygiene.readme.status, language)}
                </Badge>
              ) : null}
            </div>

            <div className="grid gap-3 lg:grid-cols-2 xl:grid-cols-4">
              <div className="rounded-xl border border-border/70 bg-surface/60 p-3">
                <div className="mb-3 flex items-center justify-between gap-2">
                  <h3 className="text-sm font-semibold text-foreground">{text.inventoryLabel}</h3>
                  <Badge
                    variant={researchStatus.dataHygiene.inventory.status === 'available' ? 'success' : 'default'}
                    size="sm"
                  >
                    <StatusDot
                      tone={researchStatus.dataHygiene.inventory.status === 'available' ? 'success' : 'neutral'}
                      className="h-1.5 w-1.5"
                    />
                    {dataStatusLabel(researchStatus.dataHygiene.inventory.status, language)}
                  </Badge>
                </div>
                {researchStatus.dataHygiene.inventory.status === 'available' ? (
                  <dl className="space-y-1 text-xs text-secondary-text">
                    <div className="flex justify-between gap-3">
                      <dt>{text.officialLabel}</dt>
                      <dd className="font-medium text-foreground">{researchStatus.dataHygiene.inventory.officialEntries ?? 0}</dd>
                    </div>
                    <div className="flex justify-between gap-3">
                      <dt>{text.candidateLabel}</dt>
                      <dd className="font-medium text-foreground">{researchStatus.dataHygiene.inventory.candidateEntries ?? 0}</dd>
                    </div>
                    <div className="flex justify-between gap-3">
                      <dt>{text.legacyLabel}</dt>
                      <dd className="font-medium text-foreground">{researchStatus.dataHygiene.inventory.legacyEntries ?? 0}</dd>
                    </div>
                    <div className="flex justify-between gap-3">
                      <dt>{text.archiveLabel}</dt>
                      <dd className="font-medium text-foreground">{researchStatus.dataHygiene.inventory.archiveCandidates ?? 0}</dd>
                    </div>
                  </dl>
                ) : (
                  <p className="text-xs text-secondary-text">{language === 'zh' ? '暂无盘点' : 'Not available'}</p>
                )}
              </div>

              <div className="rounded-xl border border-border/70 bg-surface/60 p-3">
                <div className="mb-3 flex items-center justify-between gap-2">
                  <h3 className="text-sm font-semibold text-foreground">{text.cleanupPlanLabel}</h3>
                  <Badge
                    variant={researchStatus.dataHygiene.cleanupPlan.status === 'available' ? 'success' : 'default'}
                    size="sm"
                  >
                    <StatusDot
                      tone={researchStatus.dataHygiene.cleanupPlan.status === 'available' ? 'success' : 'neutral'}
                      className="h-1.5 w-1.5"
                    />
                    {dataStatusLabel(researchStatus.dataHygiene.cleanupPlan.status, language)}
                  </Badge>
                </div>
                {researchStatus.dataHygiene.cleanupPlan.status === 'available' ? (
                  <dl className="space-y-1 text-xs text-secondary-text">
                    <div className="flex justify-between gap-3">
                      <dt>{text.actionsLabel}</dt>
                      <dd className="font-medium text-foreground">{researchStatus.dataHygiene.cleanupPlan.totalActions ?? 0}</dd>
                    </div>
                    <div className="flex justify-between gap-3">
                      <dt>{text.dryRunLabel}</dt>
                      <dd className="font-medium text-foreground">
                        {researchStatus.dataHygiene.cleanupPlan.dryRunOnly ? (language === 'zh' ? '是' : 'Yes') : (language === 'zh' ? '否' : 'No')}
                      </dd>
                    </div>
                  </dl>
                ) : (
                  <p className="text-xs text-secondary-text">{language === 'zh' ? '暂无计划' : 'Not available'}</p>
                )}
              </div>

              <div className="rounded-xl border border-border/70 bg-surface/60 p-3">
                <div className="mb-3 flex items-center justify-between gap-2">
                  <h3 className="text-sm font-semibold text-foreground">{text.qualityReportLabel}</h3>
                  <Badge
                    variant={researchStatus.dataHygiene.qualityReport.status === 'available' ? 'success' : 'default'}
                    size="sm"
                  >
                    <StatusDot
                      tone={researchStatus.dataHygiene.qualityReport.status === 'available' ? 'success' : 'neutral'}
                      className="h-1.5 w-1.5"
                    />
                    {dataStatusLabel(researchStatus.dataHygiene.qualityReport.status, language)}
                  </Badge>
                </div>
                {researchStatus.dataHygiene.qualityReport.status === 'available' ? (
                  <>
                    {researchStatus.dataHygiene.qualityReport.summary ? (
                      <dl className="space-y-1 text-xs text-secondary-text">
                        {researchStatus.dataHygiene.qualityReport.summary.stockUniverse && (
                          <div className="flex justify-between gap-3">
                            <dt>{text.stockUniverse}</dt>
                            <dd className="font-medium text-foreground">{researchStatus.dataHygiene.qualityReport.summary.stockUniverse}</dd>
                          </div>
                        )}
                        {researchStatus.dataHygiene.qualityReport.summary.usableInstruments && (
                          <div className="flex justify-between gap-3">
                            <dt>{text.usableInstruments}</dt>
                            <dd className="font-medium text-foreground">{researchStatus.dataHygiene.qualityReport.summary.usableInstruments}</dd>
                          </div>
                        )}
                        {researchStatus.dataHygiene.qualityReport.summary.totalRows && (
                          <div className="flex justify-between gap-3">
                            <dt>{text.totalRows}</dt>
                            <dd className="font-medium text-foreground">{researchStatus.dataHygiene.qualityReport.summary.totalRows}</dd>
                          </div>
                        )}
                        {researchStatus.dataHygiene.qualityReport.summary.dateRange && (
                          <div className="flex justify-between gap-3">
                            <dt>{text.dateRange}</dt>
                            <dd className="font-medium text-foreground">{researchStatus.dataHygiene.qualityReport.summary.dateRange}</dd>
                          </div>
                        )}
                      </dl>
                    ) : researchStatus.dataHygiene.qualityReport.path ? (
                      <p className="break-all text-xs text-secondary-text">{researchStatus.dataHygiene.qualityReport.path}</p>
                    ) : null}
                  </>
                ) : (
                  <p className="text-xs text-secondary-text">{language === 'zh' ? '暂无报告' : 'Not available'}</p>
                )}
              </div>

              <div className="rounded-xl border border-border/70 bg-surface/60 p-3">
                <div className="mb-3 flex items-center justify-between gap-2">
                  <h3 className="text-sm font-semibold text-foreground">{text.cleaningStrategyLabel}</h3>
                  <Badge
                    variant={researchStatus.dataHygiene.cleaningStrategy.status === 'available' ? 'success' : 'default'}
                    size="sm"
                  >
                    <StatusDot
                      tone={researchStatus.dataHygiene.cleaningStrategy.status === 'available' ? 'success' : 'neutral'}
                      className="h-1.5 w-1.5"
                    />
                    {dataStatusLabel(researchStatus.dataHygiene.cleaningStrategy.status, language)}
                  </Badge>
                </div>
                {researchStatus.dataHygiene.cleaningStrategy.status === 'available' && researchStatus.dataHygiene.cleaningStrategy.path ? (
                  <p className="break-all text-xs text-secondary-text">{researchStatus.dataHygiene.cleaningStrategy.path}</p>
                ) : (
                  <p className="text-xs text-secondary-text">{language === 'zh' ? '暂无报告' : 'Not available'}</p>
                )}
              </div>
            </div>

            {researchStatus.dataHygiene.cleanupPlan.actions && researchStatus.dataHygiene.cleanupPlan.actions.length > 0 && (
              <div className="mt-4 rounded-xl border border-border/70 bg-surface/60 p-4">
                <h3 className="mb-3 text-sm font-semibold text-foreground">{text.cleanupActionsTitle}</h3>
                <div className="space-y-2">
                  {researchStatus.dataHygiene.cleanupPlan.actions.slice(0, 10).map((action, index) => (
                    <details key={index} className="group rounded-lg border border-border/50 bg-background/50">
                      <summary className="flex cursor-pointer items-center gap-2 px-3 py-2 text-xs hover:bg-surface/50">
                        <Badge variant="default" size="sm" className="shrink-0">
                          {action.action}
                        </Badge>
                        <span className="flex-1 truncate font-mono text-secondary-text">
                          {cleanupActionSourceLabel(action.source, language)}
                        </span>
                        {action.requiresConfirmation && (
                          <Badge variant="warning" size="sm" className="shrink-0">
                            {text.confirmationRequired}
                          </Badge>
                        )}
                        <ArrowRight className="h-3 w-3 shrink-0 transition-transform group-open:rotate-90" />
                      </summary>
                      <div className="space-y-2 border-t border-border/50 px-3 py-2 text-xs">
                        <div>
                          <dt className="font-medium text-secondary-text">{text.sourceLabel}:</dt>
                          <dd className="mt-0.5 break-all font-mono text-foreground">{action.source || '-'}</dd>
                        </div>
                        {action.destination && (
                          <div>
                            <dt className="font-medium text-secondary-text">{text.destinationLabel}:</dt>
                            <dd className="mt-0.5 break-all font-mono text-foreground">{action.destination}</dd>
                          </div>
                        )}
                        <div>
                          <dt className="font-medium text-secondary-text">{text.reasonLabel}:</dt>
                          <dd className="mt-0.5 text-secondary-text">{action.reason}</dd>
                        </div>
                        {action.risk && (
                          <div>
                            <dt className="font-medium text-warning">{text.riskLabel}:</dt>
                            <dd className="mt-0.5 text-secondary-text">{action.risk}</dd>
                          </div>
                        )}
                      </div>
                    </details>
                  ))}
                  {researchStatus.dataHygiene.cleanupPlan.actions.length > 10 && (
                    <p className="pt-2 text-center text-xs text-secondary-text">
                      {language === 'zh'
                        ? `显示前 10 个动作，共 ${researchStatus.dataHygiene.cleanupPlan.actions.length} 个`
                        : `Showing first 10 of ${researchStatus.dataHygiene.cleanupPlan.actions.length} actions`}
                    </p>
                  )}
                </div>
              </div>
            )}
          </Card>
        </section>
      ) : null}

      <section aria-label={text.coverageTitle}>
        <Card padding="lg" className="space-y-4">
          <div className="flex flex-col gap-2 lg:flex-row lg:items-start lg:justify-between">
            <div>
              <h2 className="text-lg font-semibold text-foreground">{text.coverageTitle}</h2>
              <p className="mt-1 max-w-4xl text-sm leading-6 text-secondary-text">{text.coverageDescription}</p>
            </div>
            <Badge variant="history" size="sm" className="w-fit rounded-lg">
              {text.coverageTestsLabel} / {text.coverageBrowserLabel} / {text.coverageReviewLabel}
            </Badge>
          </div>

          <div className="grid gap-3 xl:hidden">
            {coverageItems.map((item) => (
              <article key={item.module} className="rounded-xl border border-border/70 bg-surface/60 p-3">
                <div className="flex flex-wrap items-start justify-between gap-2">
                  <h3 className="min-w-0 break-words text-sm font-semibold text-foreground">{item.module}</h3>
                  <StatusBadge status={item.status} language={language} />
                </div>
                <dl className="mt-3 space-y-2 text-xs leading-5 text-secondary-text">
                  <div>
                    <dt className="font-medium text-muted-text">{text.coverageEntryLabel}</dt>
                    <dd className="mt-1 break-words text-foreground">
                      {item.route ? <Link to={item.route} className="text-cyan hover:underline">{item.entry}</Link> : item.entry}
                    </dd>
                  </div>
                  <div>
                    <dt className="font-medium text-muted-text">{text.coverageTestsLabel}</dt>
                    <dd className="mt-1 break-words">{item.tests}</dd>
                  </div>
                  <div>
                    <dt className="font-medium text-muted-text">{text.coverageBrowserLabel}</dt>
                    <dd className="mt-1 break-words">{item.browser}</dd>
                  </div>
                  <div>
                    <dt className="font-medium text-muted-text">{text.coverageReviewLabel}</dt>
                    <dd className="mt-1 break-words">{item.review}</dd>
                  </div>
                  <div>
                    <dt className="font-medium text-muted-text">{text.coverageNextLabel}</dt>
                    <dd className="mt-1 break-words">{item.next}</dd>
                  </div>
                </dl>
              </article>
            ))}
          </div>

          <div className="hidden xl:block">
            <div className="grid grid-cols-[1.25fr_1.35fr_1.4fr_1.4fr_1fr_1.25fr] gap-0 overflow-hidden rounded-xl border border-border/70 bg-surface/60 text-xs">
              {[
                text.coverageModuleLabel,
                text.coverageEntryLabel,
                text.coverageTestsLabel,
                text.coverageBrowserLabel,
                text.coverageReviewLabel,
                text.coverageNextLabel,
              ].map((heading) => (
                <div key={heading} className="border-b border-border/70 bg-background/50 px-3 py-2 font-semibold text-muted-text">
                  {heading}
                </div>
              ))}
              {coverageItems.map((item) => (
                <Fragment key={item.module}>
                  <div className="border-t border-border/60 px-3 py-3">
                    <div className="flex flex-col gap-2">
                      <span className="break-words font-semibold text-foreground">{item.module}</span>
                      <StatusBadge status={item.status} language={language} />
                    </div>
                  </div>
                  <div className="border-t border-border/60 px-3 py-3 leading-5 text-secondary-text">
                    {item.route ? <Link to={item.route} className="break-words text-cyan hover:underline">{item.entry}</Link> : item.entry}
                  </div>
                  <div className="border-t border-border/60 px-3 py-3 leading-5 text-secondary-text">{item.tests}</div>
                  <div className="border-t border-border/60 px-3 py-3 leading-5 text-secondary-text">{item.browser}</div>
                  <div className="border-t border-border/60 px-3 py-3 leading-5 text-secondary-text">{item.review}</div>
                  <div className="border-t border-border/60 px-3 py-3 leading-5 text-secondary-text">{item.next}</div>
                </Fragment>
              ))}
            </div>
          </div>
        </Card>
      </section>

      <section aria-label={text.inventoryTitle}>
        <Card padding="lg" className="space-y-4">
          <div className="flex flex-col gap-2 lg:flex-row lg:items-start lg:justify-between">
            <div>
              <h2 className="text-lg font-semibold text-foreground">{text.inventoryTitle}</h2>
              <p className="mt-1 max-w-4xl text-sm leading-6 text-secondary-text">{text.inventoryDescription}</p>
            </div>
            <Badge variant="info" size="sm" className="w-fit rounded-lg">
              {formatCompactNumber(functionInventoryItems.length)}
            </Badge>
          </div>

          <div className="grid gap-3 xl:hidden">
            {functionInventoryItems.map((item) => (
              <article key={item.id} className="rounded-xl border border-border/70 bg-surface/60 p-3">
                <div className="flex flex-wrap items-start justify-between gap-2">
                  <div className="min-w-0">
                    <p className="text-xs font-medium text-muted-text">{item.area}</p>
                    <h3 className="mt-1 break-words text-sm font-semibold text-foreground">{item.name}</h3>
                  </div>
                  <StatusBadge status={normalizeSupportState(item.status)} language={language} />
                </div>
                <dl className="mt-3 space-y-2 text-xs leading-5 text-secondary-text">
                  <div>
                    <dt className="font-medium text-muted-text">{text.coverageEntryLabel}</dt>
                    <dd className="mt-1 break-words text-foreground">
                      {item.route ? <Link to={item.route} className="text-cyan hover:underline">{item.entry}</Link> : item.entry}
                    </dd>
                  </div>
                  <div>
                    <dt className="font-medium text-muted-text">{text.inventoryCoverageLabel}</dt>
                    <dd className="mt-1 break-words">{item.coverage}</dd>
                  </div>
                  <div>
                    <dt className="font-medium text-muted-text">{text.inventoryGapsLabel}</dt>
                    <dd className="mt-1 break-words">{item.gaps}</dd>
                  </div>
                  <div>
                    <dt className="font-medium text-muted-text">{text.coverageNextLabel}</dt>
                    <dd className="mt-1 break-words">{item.next}</dd>
                  </div>
                </dl>
              </article>
            ))}
          </div>

          <div className="hidden xl:block">
            <div className="grid grid-cols-[0.85fr_1fr_1.25fr_1.45fr_1.25fr_1.25fr] overflow-hidden rounded-xl border border-border/70 bg-surface/60 text-xs">
              {[
                text.inventoryAreaLabel,
                text.coverageModuleLabel,
                text.coverageEntryLabel,
                text.inventoryCoverageLabel,
                text.inventoryGapsLabel,
                text.inventoryVerificationLabel,
              ].map((heading) => (
                <div key={heading} className="border-b border-border/70 bg-background/50 px-3 py-2 font-semibold text-muted-text">
                  {heading}
                </div>
              ))}
              {functionInventoryItems.map((item) => (
                <Fragment key={item.id}>
                  <div className="border-t border-border/60 px-3 py-3 leading-5 text-secondary-text">{item.area}</div>
                  <div className="border-t border-border/60 px-3 py-3">
                    <div className="flex flex-col gap-2">
                      <span className="break-words font-semibold text-foreground">{item.name}</span>
                      <StatusBadge status={normalizeSupportState(item.status)} language={language} />
                    </div>
                  </div>
                  <div className="border-t border-border/60 px-3 py-3 leading-5 text-secondary-text">
                    {item.route ? <Link to={item.route} className="break-words text-cyan hover:underline">{item.entry}</Link> : item.entry}
                    <p className="mt-1 break-words text-muted-text">{item.owner}</p>
                  </div>
                  <div className="border-t border-border/60 px-3 py-3 leading-5 text-secondary-text">{item.coverage}</div>
                  <div className="border-t border-border/60 px-3 py-3 leading-5 text-secondary-text">{item.gaps}</div>
                  <div className="border-t border-border/60 px-3 py-3 leading-5 text-secondary-text">
                    <p>{item.verification}</p>
                    <p className="mt-2 text-muted-text">{item.next}</p>
                  </div>
                </Fragment>
              ))}
            </div>
          </div>
        </Card>
      </section>

      <section aria-label={text.chinaReuseTitle}>
        <Card padding="lg" className="space-y-4">
          <div className="flex flex-col gap-2 lg:flex-row lg:items-start lg:justify-between">
            <div>
              <h2 className="text-lg font-semibold text-foreground">{text.chinaReuseTitle}</h2>
              <p className="mt-1 max-w-4xl text-sm leading-6 text-secondary-text">{text.chinaReuseDescription}</p>
            </div>
            <Badge variant="warning" size="sm" className="w-fit rounded-lg">
              {formatCompactNumber(chinaReuseItems.length)}
            </Badge>
          </div>

          <div className="grid gap-3 lg:grid-cols-2 2xl:grid-cols-3">
            {chinaReuseItems.map((item) => {
              const restrictions = item.restrictions ?? [];
              return (
                <article key={item.id} className="flex min-h-[260px] flex-col rounded-xl border border-border/70 bg-surface/60 p-4">
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <p className="text-xs font-medium text-muted-text">{item.priority}</p>
                      <h3 className="mt-1 break-words text-base font-semibold text-foreground">{item.capability}</h3>
                    </div>
                    <StatusBadge status={normalizeSupportState(item.status)} language={language} />
                  </div>
                  <dl className="mt-4 space-y-2 text-xs leading-5 text-secondary-text">
                    <div>
                      <dt className="font-medium text-muted-text">{text.chinaReuseSourceLabel}</dt>
                      <dd className="mt-1 break-all font-mono text-foreground">{item.source}</dd>
                    </div>
                    <div>
                      <dt className="font-medium text-muted-text">{text.chinaReuseDecisionLabel}</dt>
                      <dd className="mt-1 break-words">{item.decision}</dd>
                    </div>
                    <div>
                      <dt className="font-medium text-muted-text">{text.chinaReuseTargetLabel}</dt>
                      <dd className="mt-1 break-words">{item.target}</dd>
                    </div>
                    {restrictions.length ? (
                      <div>
                        <dt className="font-medium text-muted-text">{text.chinaReuseRestrictionLabel}</dt>
                        <dd className="mt-1 break-words">{joinResearchList(restrictions, language)}</dd>
                      </div>
                    ) : null}
                  </dl>
                  <p className="mt-auto pt-4 text-sm leading-6 text-secondary-text">{item.next}</p>
                </article>
              );
            })}
          </div>
        </Card>
      </section>

      <section aria-label={text.acceptanceTitle}>
        <Card padding="lg" className="space-y-4">
          <div className="flex flex-col gap-2 lg:flex-row lg:items-start lg:justify-between">
            <div>
              <h2 className="text-lg font-semibold text-foreground">{text.acceptanceTitle}</h2>
              <p className="mt-1 max-w-4xl text-sm leading-6 text-secondary-text">{text.acceptanceDescription}</p>
            </div>
            <Badge variant="success" size="sm" className="w-fit rounded-lg">
              {formatCompactNumber(acceptanceItems.length)}
            </Badge>
          </div>

          <div className="grid gap-3 lg:grid-cols-2 2xl:grid-cols-3">
            {acceptanceItems.map((item) => (
              <article key={item.id} className="flex min-h-[210px] flex-col rounded-xl border border-border/70 bg-surface/60 p-4">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <p className="break-all font-mono text-xs text-muted-text">{item.id}</p>
                    <h3 className="mt-1 break-words text-base font-semibold text-foreground">
                      {item.route ? <Link to={item.route} className="hover:text-cyan hover:underline">{item.title}</Link> : item.title}
                    </h3>
                  </div>
                  <StatusBadge status={normalizeSupportState(item.status)} language={language} />
                </div>
                <dl className="mt-4 space-y-3 text-xs leading-5 text-secondary-text">
                  <div>
                    <dt className="font-medium text-muted-text">{text.acceptanceEvidenceLabel}</dt>
                    <dd className="mt-1 break-words">{item.evidence}</dd>
                  </div>
                  <div>
                    <dt className="font-medium text-muted-text">{text.acceptanceNextLabel}</dt>
                    <dd className="mt-1 break-words">{item.next}</dd>
                  </div>
                </dl>
              </article>
            ))}
          </div>
        </Card>
      </section>

      <section aria-label={text.developmentQueueTitle}>
        <Card padding="lg" className="space-y-4">
          <div className="flex flex-col gap-2 lg:flex-row lg:items-start lg:justify-between">
            <div>
              <h2 className="text-lg font-semibold text-foreground">{text.developmentQueueTitle}</h2>
              <p className="mt-1 max-w-4xl text-sm leading-6 text-secondary-text">{text.developmentQueueDescription}</p>
            </div>
            <Badge variant="info" size="sm" className="w-fit rounded-lg">
              {text.developmentPriorityLabel}
            </Badge>
          </div>

          <div className="divide-y divide-border/70 overflow-hidden rounded-xl border border-border/70 bg-surface/60">
            {text.developmentQueue.map((item) => (
              <div key={`${item.priority}-${item.title}`} className="grid gap-3 px-3 py-3 text-sm lg:grid-cols-[84px_minmax(0,1.2fr)_minmax(0,1fr)_minmax(0,1.35fr)_minmax(0,1.35fr)] lg:items-start">
                <div className="flex items-center gap-2">
                  <Badge variant="history" size="sm" className="rounded-lg">{item.priority}</Badge>
                  <StatusBadge status={item.status} language={language} />
                </div>
                <div className="min-w-0">
                  <p className="font-semibold text-foreground">{item.title}</p>
                  <p className="mt-1 text-xs font-medium text-muted-text lg:hidden">{text.developmentEntryLabel}</p>
                  <p className="mt-1 text-xs leading-5 text-secondary-text lg:hidden">
                    {item.route ? <Link to={item.route} className="text-cyan hover:underline">{item.entry}</Link> : item.entry}
                  </p>
                </div>
                <div className="min-w-0 lg:hidden">
                  <p className="text-xs font-medium text-muted-text">{text.developmentActionLabel}</p>
                  <p className="mt-1 text-xs leading-5 text-secondary-text">{item.next}</p>
                </div>
                <div className="hidden min-w-0 lg:block">
                  <p className="text-xs font-medium text-muted-text">{text.developmentEntryLabel}</p>
                  <p className="mt-1 text-xs leading-5 text-secondary-text">
                    {item.route ? <Link to={item.route} className="text-cyan hover:underline">{item.entry}</Link> : item.entry}
                  </p>
                </div>
                <div className="hidden min-w-0 lg:block">
                  <p className="text-xs font-medium text-muted-text">{text.developmentActionLabel}</p>
                  <p className="mt-1 text-xs leading-5 text-secondary-text">{item.next}</p>
                </div>
                <div className="min-w-0">
                  <p className="text-xs font-medium text-muted-text">{text.developmentVerificationLabel}</p>
                  <p className="mt-1 text-xs leading-5 text-secondary-text">{item.verification}</p>
                </div>
              </div>
            ))}
          </div>
        </Card>
      </section>

      <section aria-label={text.experimentNotebookTitle}>
        <Card padding="lg" className="space-y-5">
          <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
            <div>
              <h2 className="text-lg font-semibold text-foreground">{text.experimentNotebookTitle}</h2>
              <p className="mt-1 max-w-3xl text-sm leading-6 text-secondary-text">
                {text.experimentNotebookDescription}
              </p>
              {experimentNotebook?.logPath ? (
                <p className="mt-2 break-all text-xs text-muted-text">
                  {text.notebookFileLabel}: {experimentNotebook.logPath}
                </p>
              ) : null}
              {deleteError ? (
                <p className="mt-2 text-xs text-danger">
                  {text.deleteExperimentFailed}: {deleteError}
                </p>
              ) : null}
            </div>
            <Badge variant={notebookStatusVariant} size="sm" className="w-fit">
              <StatusDot tone={notebookStatusTone} className="h-1.5 w-1.5" />
              {dataStatusLabel(notebookStatus, language)}
            </Badge>
          </div>

          <div className="rounded-xl border border-cyan/20 bg-cyan/5 p-4">
            <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
              <div>
                <h3 className="text-base font-semibold text-foreground">{text.experimentComparisonTitle}</h3>
                <p className="mt-1 text-sm leading-6 text-secondary-text">{text.experimentComparisonDescription}</p>
              </div>
              <Badge variant="history" size="sm" className="w-fit rounded-lg">
                {text.comparisonMetricLabel}
              </Badge>
            </div>

            {experimentComparison ? (
              <div className="mt-4 grid gap-3 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_160px]">
                <div className="rounded-xl border border-border/70 bg-background/60 p-3">
                  <p className="text-xs font-medium text-muted-text">{text.cleanedDataLabel}</p>
                  <p className={`mt-2 text-lg font-semibold ${marketMetricClass(experimentComparison.cleanedValue)}`}>
                    {formatMetricValue(experimentComparison.cleanedValue)}
                  </p>
                  <p className="mt-1 break-words text-xs text-secondary-text">
                    {experimentComparison.cleanedRun.experimentName ?? experimentComparison.cleanedRun.runId}
                  </p>
                </div>
                <div className="rounded-xl border border-border/70 bg-background/60 p-3">
                  <p className="text-xs font-medium text-muted-text">{text.canonicalDataLabel}</p>
                  <p className="mt-2 text-lg font-semibold text-foreground">
                    {formatMetricValue(experimentComparison.canonicalValue)}
                  </p>
                  <p className="mt-1 break-words text-xs text-secondary-text">
                    {experimentComparison.canonicalRun.experimentName ?? experimentComparison.canonicalRun.runId}
                  </p>
                </div>
                <div className="rounded-xl border border-border/70 bg-background/60 p-3">
                  <p className="text-xs font-medium text-muted-text">{text.comparisonDeltaLabel}</p>
                  <p className={`mt-2 text-lg font-semibold ${marketMetricClass(experimentComparison.delta)}`}>
                    {formatSignedMetricValue(experimentComparison.delta)}
                  </p>
                  <p className="mt-1 break-all font-mono text-xs text-secondary-text">
                    {experimentComparison.metricKey}
                  </p>
                </div>
              </div>
            ) : (
              <p className="mt-4 rounded-xl border border-border/70 bg-background/60 p-3 text-sm leading-6 text-secondary-text">
                {text.comparisonNoData}
              </p>
            )}
          </div>

          <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_430px]">
            <div className="rounded-xl border border-border/70 bg-surface/60 p-4">
              <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                <h3 className="text-base font-semibold text-foreground">{text.latestNoteTitle}</h3>
                <Badge variant="history" size="sm" className="rounded-lg">
                  {experimentCountLabel}
                </Badge>
              </div>
              {experimentNotebook?.latestNote ? (
                <div className="max-h-72 overflow-auto whitespace-pre-wrap rounded-xl border border-border/70 bg-background/60 p-3 text-sm leading-6 text-secondary-text">
                  {experimentNotebook.latestNote}
                </div>
              ) : (
                <p className="rounded-xl border border-border/70 bg-background/60 p-3 text-sm leading-6 text-secondary-text">
                  {text.noExperimentNote}
                </p>
              )}
            </div>

            <div className="rounded-xl border border-border/70 bg-surface/60 p-4">
              <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                <h3 className="text-base font-semibold text-foreground">{text.latestRunsTitle}</h3>
                <Badge variant="default" size="sm" className="rounded-lg">
                  {formatCompactNumber(experimentRunCount)}
                </Badge>
              </div>
              {experimentRuns.length > 0 ? (
                <div className="space-y-3">
                  {experimentRuns.slice(0, 3).map((run, index) => (
                    <article key={run.runId ?? `${run.experimentName}-${index}`} className="rounded-xl border border-border/70 bg-background/60 p-3">
                      <div className="flex items-start justify-between gap-3">
                        <div className="min-w-0">
                          <h4 className="break-words text-sm font-semibold text-foreground">
                            {run.experimentName ?? run.status}
                          </h4>
                          {run.runId ? (
                            <p className="mt-1 break-all font-mono text-xs text-muted-text">
                              {text.runIdLabel} {run.runId}
                            </p>
                          ) : null}
                        </div>
                        <div className="flex shrink-0 flex-wrap items-center justify-end gap-2">
                          <Badge variant={run.status === 'completed' ? 'success' : 'default'} size="sm">
                            {run.status}
                          </Badge>
                          {run.runId ? (
                            <>
                              <button
                                type="button"
                                aria-label={`${text.archiveExperimentAria} ${run.runId}`}
                                disabled={mutatingRunKey === `archive:${run.runId}` || mutatingRunKey === `permanent:${run.runId}`}
                                onClick={() => { void handleArchiveExperiment(run.runId as string); }}
                                className="inline-flex h-8 items-center gap-1.5 rounded-lg border border-warning/30 px-2 text-xs font-medium text-warning transition-colors hover:bg-warning/10 disabled:cursor-not-allowed disabled:opacity-60"
                              >
                                <Archive className="h-3.5 w-3.5" />
                                {text.archiveExperiment}
                              </button>
                              <button
                                type="button"
                                aria-label={`${text.permanentDeleteExperimentAria} ${run.runId}`}
                                disabled={mutatingRunKey === `archive:${run.runId}` || mutatingRunKey === `permanent:${run.runId}`}
                                onClick={() => { void handlePermanentDeleteExperiment(run.runId as string); }}
                                className="inline-flex h-8 items-center gap-1.5 rounded-lg border border-danger/30 px-2 text-xs font-medium text-danger transition-colors hover:bg-danger/10 disabled:cursor-not-allowed disabled:opacity-60"
                              >
                                <Trash2 className="h-3.5 w-3.5" />
                                {text.permanentDeleteExperiment}
                              </button>
                            </>
                          ) : null}
                        </div>
                      </div>

                      <dl className="mt-3 space-y-1 text-xs text-secondary-text">
                        {run.model ? (
                          <div className="flex justify-between gap-3">
                            <dt>{text.modelLabel}</dt>
                            <dd className="font-medium text-foreground">{run.model}</dd>
                          </div>
                        ) : null}
                        {run.strategy ? (
                          <div className="flex justify-between gap-3">
                            <dt>{text.strategyLabel}</dt>
                            <dd className="font-medium text-foreground">{run.strategy}</dd>
                          </div>
                        ) : null}
                        {run.datasetRange ? (
                          <div className="flex justify-between gap-3">
                            <dt>{text.datasetLabel}</dt>
                            <dd className="text-right font-medium text-foreground">{run.datasetRange}</dd>
                          </div>
                        ) : null}
                      </dl>

                      {run.conclusion ? (
                        <p className="mt-3 text-xs leading-5 text-secondary-text">
                          <span className="font-medium text-foreground">{text.conclusionLabel}: </span>
                          {run.conclusion}
                        </p>
                      ) : null}
                      {run.nextStep ? (
                        <p className="mt-2 text-xs leading-5 text-secondary-text">
                          <span className="font-medium text-foreground">{text.nextStepLabel}: </span>
                          {run.nextStep}
                        </p>
                      ) : null}
                      <div className="mt-3 flex flex-wrap gap-2">
                        {Object.entries(run.metrics ?? {}).slice(0, 2).map(([name, value]) => (
                          <Badge key={name} variant="history" size="sm" className="rounded-lg">
                            {name}: {formatMetricValue(value)}
                          </Badge>
                        ))}
                      </div>
                    </article>
                  ))}
                </div>
              ) : (
                <p className="rounded-xl border border-border/70 bg-background/60 p-3 text-sm leading-6 text-secondary-text">
                  {text.noExperimentRuns}
                </p>
              )}
            </div>
          </div>
        </Card>
      </section>

      <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_390px]">
        <div className="space-y-5">
          <section>
            <div className="mb-3 flex flex-col gap-1 sm:flex-row sm:items-end sm:justify-between">
              <div>
                <h2 className="text-lg font-semibold text-foreground">{text.capabilitiesTitle}</h2>
                <p className="mt-1 max-w-3xl text-sm text-secondary-text">{text.capabilitiesDescription}</p>
              </div>
            </div>

            <div className="grid gap-3 md:grid-cols-2 2xl:grid-cols-3">
              {text.capabilities.map((item) => {
                const Icon = item.icon;
                return (
                  <Card key={item.title} padding="md" hoverable className="flex min-h-[252px] flex-col">
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex min-w-0 items-center gap-3">
                        <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-cyan/20 bg-cyan/10 text-cyan">
                          <Icon className="h-5 w-5" />
                        </span>
                        <h3 className="text-base font-semibold text-foreground">{item.title}</h3>
                      </div>
                      <StatusBadge status={item.status} language={language} />
                    </div>

                    <p className="mt-4 text-sm leading-6 text-secondary-text">{item.description}</p>
                    <div className="mt-4 flex flex-wrap gap-2">
                      {item.focus.map((focus) => (
                        <Badge key={focus} variant="default" size="sm" className="rounded-lg">
                          {focus}
                        </Badge>
                      ))}
                    </div>
                    <div className="mt-auto pt-5">
                      <EntryLink to={item.route}>{item.routeLabel}</EntryLink>
                    </div>
                  </Card>
                );
              })}
            </div>
          </section>

          <section className="grid gap-4 lg:grid-cols-2">
            {text.modes.map((item) => {
              const Icon = item.icon;
              return (
                <Card key={item.title} padding="lg" variant={item.title.includes('ETF') || item.title.includes('基金') ? 'gradient' : 'default'}>
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex items-center gap-3">
                      <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl border border-purple/20 bg-purple/10 text-purple">
                        <Icon className="h-5 w-5" />
                      </span>
                      <div>
                        <p className="label-uppercase">{text.dualModeTitle}</p>
                        <h2 className="mt-1 text-lg font-semibold text-foreground">{item.title}</h2>
                      </div>
                    </div>
                    <StatusBadge status={item.status} language={language} />
                  </div>
                  <p className="mt-4 text-sm leading-6 text-secondary-text">{item.description}</p>
                  <ul className="mt-4 space-y-2 text-sm text-secondary-text">
                    {item.points.map((point) => (
                      <li key={point} className="flex gap-2 leading-6">
                        <StatusDot tone={item.status === 'available' ? 'success' : 'warning'} className="mt-2 h-1.5 w-1.5" />
                        <span>{point}</span>
                      </li>
                    ))}
                  </ul>
                </Card>
              );
            })}
          </section>

          <section>
            <div className="mb-3">
              <h2 className="text-lg font-semibold text-foreground">{text.etfBoundaryTitle}</h2>
              <p className="mt-1 max-w-3xl text-sm text-secondary-text">{text.dualModeDescription}</p>
            </div>
            <div className="grid gap-3 md:grid-cols-3">
              {text.etfBoundaries.map((item) => (
                <Card key={item.title} padding="md" className="min-h-[158px]">
                  <div className="flex items-center justify-between gap-3">
                    <h3 className="text-base font-semibold text-foreground">{item.title}</h3>
                    <StatusBadge status={item.status} language={language} />
                  </div>
                  <p className="mt-4 text-sm leading-6 text-secondary-text">{item.detail}</p>
                </Card>
              ))}
            </div>
          </section>

          <section>
            <div className="mb-3">
              <h2 className="text-lg font-semibold text-foreground">{text.futureTitle}</h2>
              <p className="mt-1 max-w-3xl text-sm text-secondary-text">{text.futureDescription}</p>
            </div>
            <div className="grid gap-3 md:grid-cols-2">
              {text.futureItems.map((item) => {
                const Icon = item.icon;
                return (
                  <Card key={item.title} padding="md">
                    <div className="flex items-start gap-3">
                      <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-warning/20 bg-warning/10 text-warning">
                        <Icon className="h-5 w-5" />
                      </span>
                      <div className="min-w-0">
                        <div className="flex flex-wrap items-center gap-2">
                          <h3 className="text-base font-semibold text-foreground">{item.title}</h3>
                          <Badge variant="warning" size="sm">{item.status}</Badge>
                        </div>
                        <p className="mt-2 text-sm leading-6 text-secondary-text">{item.description}</p>
                      </div>
                    </div>
                  </Card>
                );
              })}
            </div>
          </section>
        </div>

        <aside className="space-y-4 xl:sticky xl:top-4 xl:self-start" aria-label={text.projectMapTitle}>
          <Card padding="lg" variant="gradient">
            <div className="mb-5">
              <p className="label-uppercase">{text.projectMapTitle}</p>
              <h2 className="mt-1 text-lg font-semibold text-foreground">{text.projectMapTitle}</h2>
              <p className="mt-2 text-sm leading-6 text-secondary-text">{text.projectMapDescription}</p>
            </div>
            <div className="space-y-3">
              {text.projectMap.map((item, index) => {
                const Icon = item.icon;
                return (
                  <div key={item.name} className="relative">
                    {index < text.projectMap.length - 1 ? (
                      <span className="absolute left-5 top-12 h-[calc(100%-1rem)] w-px bg-border" aria-hidden />
                    ) : null}
                    <div className="flex gap-3 rounded-xl border border-border/70 bg-surface/70 p-3">
                      <span className="relative z-10 flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-cyan/20 bg-background text-cyan">
                        <Icon className="h-5 w-5" />
                      </span>
                      <div className="min-w-0">
                        <h3 className="break-words text-sm font-semibold text-foreground">{item.name}</h3>
                        <p className="mt-1 text-xs font-medium text-cyan">{item.role}</p>
                        <p className="mt-2 text-xs leading-5 text-secondary-text">{item.detail}</p>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </Card>

          <Card padding="md">
            <div className="flex items-start gap-3">
              <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-cyan/20 bg-cyan/10 text-cyan">
                <Globe2 className="h-5 w-5" />
              </span>
              <div className="min-w-0">
                <p className="label-uppercase">{text.externalToolsTitle}</p>
                <h2 className="mt-1 text-base font-semibold text-foreground">{text.catdeskTitle}</h2>
                <p className="mt-2 text-sm leading-6 text-secondary-text">{text.catdeskDescription}</p>
                <p className="mt-2 text-xs leading-5 text-muted-text">{text.catdeskNote}</p>
                <a
                  href={CATDESK_URL}
                  target="_blank"
                  rel="noreferrer"
                  className="mt-3 inline-flex h-9 items-center gap-2 rounded-lg border border-cyan/30 bg-cyan/10 px-3 text-sm font-medium text-cyan transition-colors hover:bg-cyan/15"
                >
                  <ArrowUpRight className="h-4 w-4" />
                  {text.catdeskOpen}
                </a>
              </div>
            </div>
          </Card>

          <Card padding="md">
            <div className="flex items-start gap-3">
              <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-info/20 bg-cyan/10 text-cyan">
                <BarChart3 className="h-5 w-5" />
              </span>
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <h2 className="text-base font-semibold text-foreground">MLflow</h2>
                  <Badge variant={mlflowStatusMeta.variant} size="sm" className="gap-1.5 shadow-none">
                    <StatusDot tone={mlflowStatusMeta.tone} className="h-1.5 w-1.5" />
                    {mlflowStatusLabel}
                  </Badge>
                </div>
                <p className="mt-2 text-sm leading-6 text-secondary-text">{text.mlflowNote}</p>
                {researchStatus?.mlflow?.error ? (
                  <p className="mt-2 text-xs leading-5 text-muted-text">{researchStatus.mlflow.error}</p>
                ) : null}
                <p className="mt-3 rounded-xl border border-border/70 bg-surface/70 px-3 py-2 font-mono text-xs text-secondary-text">
                  {mlflowUrl}
                </p>
              </div>
            </div>
          </Card>

          <Card padding="md">
            <div className="flex items-start gap-3">
              <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-warning/20 bg-warning/10 text-warning">
                <Bell className="h-5 w-5" />
              </span>
              <p className="text-sm leading-6 text-secondary-text">{text.portNote}</p>
            </div>
          </Card>
        </aside>
      </div>
      <ConfirmDialog
        isOpen={confirmDeleteRunId !== null}
        title={text.permanentDeleteExperiment}
        message={text.permanentDeleteConfirm}
        isDanger
        onConfirm={handleConfirmPermanentDelete}
        onCancel={handleCancelPermanentDelete}
      />
    </AppPage>
  );
};

export default ResearchPage;
