import type React from 'react';
import { useCallback, useState } from 'react';
import {
  ArrowDown,
  ArrowLeft,
  ArrowUp,
  ArrowUpRight,
  BarChart3,
  BookOpen,
  BriefcaseBusiness,
  ChevronRight,
  Database,
  FlaskConical,
  LoaderCircle,
  Search,
  Settings2,
  Sparkles,
} from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import { Link } from 'react-router-dom';
import { quantApi, type LocalServiceName } from '../api/quant';
import { AppPage } from '../components/common';
import { useUiLanguage } from '../contexts/UiLanguageContext';

const LOCAL_SERVICE_DESTINATIONS: Record<LocalServiceName, string> = {
  catdesk: 'http://127.0.0.1:8765/',
  mlflow: '/experiments/',
};

const LOCAL_SERVICE_LABELS: Record<LocalServiceName, string> = {
  catdesk: '九点猫研',
  mlflow: 'MLflow',
};

const LOCAL_SERVICE_POLL_INTERVAL_MS = 400;
const LOCAL_SERVICE_START_TIMEOUT_MS = 30_000;

type LocalServiceUiStatus = 'idle' | 'checking' | 'starting' | 'failed';

type LocalServiceUiState = {
  status: LocalServiceUiStatus;
  message?: string;
};

type LandingLink = {
  name: string;
  detail: string;
  href: string;
  icon: LucideIcon;
  fullPage?: boolean;
  external?: boolean;
  service?: LocalServiceName;
};

type ProjectNode = {
  name: string;
  role: string;
  detail: string;
  icon: LucideIcon;
};

const LinkRow: React.FC<{
  item: LandingLink;
  serviceState?: LocalServiceUiState;
  onServiceClick?: (service: LocalServiceName) => void;
}> = ({ item, serviceState, onServiceClick }) => {
  const Icon = item.icon;
  const serviceStatus = item.service ? serviceState?.status ?? 'idle' : 'idle';
  const serviceBusy = serviceStatus === 'checking' || serviceStatus === 'starting';
  const detail = item.service && serviceStatus === 'checking'
      ? '正在检查服务'
      : item.service && serviceStatus === 'starting'
        ? '正在启动'
        : item.service && serviceStatus === 'failed'
        ? serviceState?.message || '启动失败，点击重试'
        : item.detail;
  const content = (
    <>
      <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-border/70 bg-surface text-cyan">
        {serviceBusy ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <Icon className="h-4 w-4" />}
      </span>
      <span className="min-w-0 flex-1">
        <span className="block text-sm font-medium text-foreground">{item.name}</span>
        <span aria-live={item.service ? 'polite' : undefined} className="mt-0.5 block text-xs text-secondary-text">{detail}</span>
      </span>
      {serviceBusy
        ? <LoaderCircle className="h-4 w-4 shrink-0 animate-spin text-cyan" />
        : item.external || item.fullPage
        ? <ArrowUpRight className="h-4 w-4 shrink-0 text-muted-text" />
        : <ChevronRight className="h-4 w-4 shrink-0 text-muted-text" />}
    </>
  );
  const className = 'flex min-h-[68px] items-center gap-3 px-4 py-3 transition-colors hover:bg-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-cyan/50';

  if (item.service) {
    return (
      <button
        type="button"
        className={`${className} w-full text-left disabled:cursor-wait disabled:opacity-80`}
        disabled={serviceBusy}
        aria-busy={serviceBusy}
        onClick={() => onServiceClick?.(item.service!)}
      >
        {content}
      </button>
    );
  }

  if (item.external || item.fullPage) {
    return (
      <a
        href={item.href}
        className={className}
        target={item.external ? '_blank' : undefined}
        rel={item.external ? 'noreferrer' : undefined}
      >
        {content}
      </a>
    );
  }

  return <Link to={item.href} className={className}>{content}</Link>;
};

function serviceErrorMessage(service: LocalServiceName, error: unknown): string {
  const responseDetail = (
    error as { response?: { data?: { detail?: unknown } } }
  ).response?.data?.detail;
  if (typeof responseDetail === 'string' && responseDetail.trim()) {
    return responseDetail;
  }
  if (responseDetail && typeof responseDetail === 'object') {
    const message = (responseDetail as { message?: unknown }).message;
    if (typeof message === 'string' && message.trim()) {
      return message;
    }
  }
  return `${LOCAL_SERVICE_LABELS[service]} 启动失败，请检查 DSA 日志。`;
}

async function waitForLocalService(service: LocalServiceName): Promise<void> {
  const deadline = Date.now() + LOCAL_SERVICE_START_TIMEOUT_MS;
  while (Date.now() < deadline) {
    const status = await quantApi.getLocalServiceStatus(service);
    if (status.status === 'available') {
      return;
    }
    await new Promise((resolve) => window.setTimeout(resolve, LOCAL_SERVICE_POLL_INTERVAL_MS));
  }
  throw new Error(`${LOCAL_SERVICE_LABELS[service]} 启动超时，请检查 DSA 日志。`);
}

const ProjectNodeCard: React.FC<{ project: ProjectNode }> = ({ project }) => {
  const Icon = project.icon;

  return (
    <article className="flex min-h-[112px] items-start gap-3 rounded-lg border border-border/70 bg-card px-4 py-4">
      <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-border/70 bg-surface text-cyan">
        <Icon className="h-4 w-4" />
      </span>
      <div className="min-w-0">
        <h3 className="break-words text-sm font-semibold text-foreground">{project.name}</h3>
        <p className="mt-1 text-xs font-medium text-cyan">{project.role}</p>
        <p className="mt-1.5 text-xs leading-5 text-secondary-text">{project.detail}</p>
      </div>
    </article>
  );
};

const VerticalRelationship: React.FC<{ label: string }> = ({ label }) => (
  <div className="flex min-h-14 items-center gap-3 pl-5 text-xs text-secondary-text">
    <ArrowDown className="h-4 w-4 shrink-0 text-cyan" />
    <span>{label}</span>
  </div>
);

const LandingPage: React.FC = () => {
  const { language } = useUiLanguage();
  const zh = language === 'zh';
  const [serviceStates, setServiceStates] = useState<Record<LocalServiceName, LocalServiceUiState>>({
    catdesk: { status: 'idle' },
    mlflow: { status: 'idle' },
  });
  const handleServiceClick = useCallback(async (service: LocalServiceName) => {
    setServiceStates((current) => ({ ...current, [service]: { status: 'checking' } }));
    try {
      const currentStatus = await quantApi.getLocalServiceStatus(service);
      if (currentStatus.status !== 'available') {
        setServiceStates((current) => ({ ...current, [service]: { status: 'starting' } }));
        const startResult = await quantApi.startLocalService(service);
        if (startResult.status === 'failed') {
          throw new Error(startResult.message || `${LOCAL_SERVICE_LABELS[service]} 启动失败。`);
        }
        if (startResult.status !== 'available') {
          await waitForLocalService(service);
        }
      }
      window.open(LOCAL_SERVICE_DESTINATIONS[service], '_self');
    } catch (error) {
      setServiceStates((current) => ({
        ...current,
        [service]: { status: 'failed', message: serviceErrorMessage(service, error) },
      }));
    }
  }, []);
  const destinations: LandingLink[] = [
    { name: zh ? '量化研究' : 'Quant Research', detail: zh ? '选股研究与交易策略研究' : 'Stock selection and trading strategies', href: '/research', icon: FlaskConical },
    { name: zh ? '个股分析' : 'Stock Analysis', detail: zh ? '个股分析与历史报告' : 'Stock analysis and reports', href: '/stock-analysis', icon: Search },
    { name: zh ? '数据状态' : 'Data Status', detail: 'quant_data', href: '/data', icon: Database },
    { name: zh ? '持仓' : 'Portfolio', detail: zh ? '组合与风险' : 'Positions and risk', href: '/portfolio', icon: BriefcaseBusiness },
    { name: '九点猫研', detail: zh ? '本地研究工具' : 'Local research tool', href: LOCAL_SERVICE_DESTINATIONS.catdesk, icon: Sparkles, service: 'catdesk' },
    { name: 'MLflow', detail: zh ? '实验记录' : 'Experiment records', href: LOCAL_SERVICE_DESTINATIONS.mlflow, icon: BarChart3, service: 'mlflow' },
    { name: zh ? '设置' : 'Settings', detail: zh ? '系统与模型配置' : 'System and model configuration', href: '/settings', icon: Settings2 },
  ];
  const dataProject: ProjectNode = {
    name: 'quant_data',
    role: zh ? '独立数据项目与服务' : 'Independent data project and service',
    detail: zh
      ? '负责下载、清洗、验收并发布不可变数据快照；不负责模型训练和 DSA 页面。'
      : 'Downloads, validates, and publishes immutable snapshots; it does not train models or serve the DSA UI.',
    icon: Database,
  };
  const workflowProject: ProjectNode = {
    name: 'quant_workflow + Qlib',
    role: zh ? 'DSA 仓内量化研究执行层' : 'Quant research layer inside the DSA repository',
    detail: zh
      ? 'quant_workflow 位于 DSA 仓库内并调用 Qlib，消费 quant_data 已发布快照执行训练、实验与回测；不发布正式数据。'
      : 'Consumes published quant_data snapshots for factors, training, experiments, and backtests; it does not publish official data.',
    icon: FlaskConical,
  };
  const dsaProject: ProjectNode = {
    name: 'daily_stock_analysis',
    role: zh ? '统一应用与展示层' : 'Unified application and presentation layer',
    detail: zh
      ? '提供 WebUI、API、研究入口、运行管理和结果展示；不连接券商自动下单。'
      : 'Provides the Web UI, API, research entry points, run management, and results; it does not place broker orders.',
    icon: BarChart3,
  };
  const mlflowProject: ProjectNode = {
    name: 'MLflow',
    role: zh ? '实验记录与产物存储' : 'Experiment tracking and artifact storage',
    detail: zh
      ? '记录 Run、参数、指标、模型与 Signal 等产物；不负责数据获取、模型训练、生成 Signal、回测或 DSA 页面。'
      : 'Records runs, parameters, metrics, and model/signal artifacts; it does not fetch data, train models, generate signals, backtest, or serve the DSA UI.',
    icon: FlaskConical,
  };
  const referenceProject: ProjectNode = {
    name: 'china-astock-quant',
    role: zh ? '独立可选策略参考库' : 'Independent optional strategy reference',
    detail: zh
      ? '提供指标、策略和工程经验供研究层按需借鉴；不是主应用，也不参与正式数据发布。'
      : 'Offers indicators, strategies, and engineering references; it is neither the main app nor an official data publisher.',
    icon: BookOpen,
  };

  return (
    <AppPage className="max-w-5xl space-y-5" >
      <header data-testid="landing-page" className="rounded-lg border border-border/70 bg-card px-5 py-5">
        <h1 className="text-2xl font-semibold text-foreground">Daily Stock Analysis</h1>
      </header>

      <section className="overflow-hidden rounded-lg border border-border/70 bg-card">
        <h2 className="border-b border-border/70 px-4 py-3 text-sm font-semibold text-foreground">
          {zh ? '常用入口' : 'Destinations'}
        </h2>
        <div className="grid divide-y divide-border/60 md:grid-cols-2 md:divide-y-0">
          {destinations.map((item, index) => (
            <div key={item.name} className={index > 1 ? 'border-t border-border/60' : ''}>
              <LinkRow
                item={item}
                serviceState={item.service ? serviceStates[item.service] : undefined}
                onServiceClick={handleServiceClick}
              />
            </div>
          ))}
        </div>
      </section>

      <section data-testid="project-relationship" className="space-y-4">
        <header className="border-b border-border/70 pb-3">
          <h2 className="text-sm font-semibold text-foreground">
            {zh ? '项目关系图' : 'Project relationships'}
          </h2>
          <p className="mt-1 text-xs text-secondary-text">
            {zh ? '数据、研究执行与应用展示各自独立，按以下方向协作。' : 'Data, research execution, and application presentation remain separate and collaborate as shown.'}
          </p>
        </header>

        <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(100px,0.32fr)_minmax(0,0.9fr)] lg:items-center">
          <div>
            <ProjectNodeCard project={dataProject} />
            <VerticalRelationship label={zh ? '提供已验收数据快照' : 'Provides approved data snapshots'} />
            <ProjectNodeCard project={workflowProject} />
            <VerticalRelationship label={zh ? '记录 Run、参数、指标与产物' : 'Records runs, parameters, metrics, and artifacts'} />
            <ProjectNodeCard project={mlflowProject} />
            <VerticalRelationship label={zh ? 'DSA 聚合展示研究与结果' : 'DSA aggregates and presents research results'} />
            <ProjectNodeCard project={dsaProject} />
          </div>

          <div className="flex items-center justify-center gap-2 text-xs text-secondary-text">
            <ArrowUp className="h-4 w-4 text-cyan lg:hidden" />
            <ArrowLeft className="hidden h-4 w-4 text-cyan lg:block" />
            <span>{zh ? '可选参考' : 'Optional reference'}</span>
          </div>

          <ProjectNodeCard project={referenceProject} />
        </div>
      </section>
    </AppPage>
  );
};

export default LandingPage;
