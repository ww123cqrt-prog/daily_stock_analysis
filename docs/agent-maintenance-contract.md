# Agent 可维护契约

本文面向 Codex、Claude Code、opencode、Copilot Coding Agent 等 agentic worker。它不是普通用户说明，而是本地量化工作台的维护契约：任何 agent 在修改 DSA 本地量化工作台、`quant_workflow` 接入、Qlib/MLflow 展示、数据治理或安装文档前，都应先阅读本文和根目录 `AGENTS.md`。

目标是让维护过程可追溯、可复现、可回滚，并且不泄露本机凭据、数据和实验产物。

## 真源关系

| 类型 | 真源 | 说明 |
| --- | --- | --- |
| 仓库协作规则 | `AGENTS.md` | 所有 agent 的最高优先级仓库规则 |
| 本地量化架构 | `docs/local-quant-workbench.md` | DSA、`quant_workflow`、Qlib、MLflow、china-astock-quant 的职责边界 |
| 安装复现 | `docs/install-local-quant-workbench.md` | 从零安装、启动服务、验证工作台 |
| 用户入口 | `docs/INDEX.md` | 人类用户文档导航 |
| 版本留痕 | `docs/CHANGELOG.md` | 用户可见变化、破坏性变化、安全边界变化 |
| 可执行验收 | `scripts/verify-local-quant-workbench.sh` | `quick`、`full`、`complete` 验收入口 |

如果文档和可执行代码冲突，先信任代码、测试和脚本，再同步修正文档。不要让文档继续漂移。

## 项目边界

当前组织决策是：组织上统一，代码上分层，数据上外置，入口统一到 DSA。不要为了“看起来整齐”把所有项目硬合成一个仓库；只有当某段能力已经稳定、可测试、可复用，并且能降低维护成本时，才把它提升为 DSA 仓内模块或共享包。

`daily_stock_analysis` 是主项目和本地网页控制台。用户通过它访问研究、工作流、回测、实验记录和设置。

仓内 `quant_workflow/` 是脚本真源。网页任务只能调用白名单脚本，不能从浏览器提交任意 shell。运行态 reports/mlruns 是本地资产：默认放在仓内 `quant_workflow/reports` 和 `quant_workflow/mlruns`，且不提交到 Git。旧安装如果仍有 DSA 同级 `quant_workflow/reports` 或 `mlruns`，路径 helper 不再按 marker 自动兜底；迁移旧历史时应复制到仓内目录，或由用户明确设置 `QUANT_WORKFLOW_REPORTS_ROOT` / `QUANT_WORKFLOW_MLRUNS_DIR` 做临时回滚。日常维护不得继续向同级旧目录写新产物。

Qlib 是通过 Python 包安装的量化研究引擎。除非任务明确要求开发 Qlib 本身，不要修改官方 Qlib 源码，也不要把 Qlib 仓库当作主 UI。

MLflow 是独立实验记录服务，不属于 Qlib，也不是 DSA 的替代 UI。DSA 只链接或读取本地 MLflow 产物摘要。

`china-astock-quant` 是 A 股指标、策略和工程经验参考库。默认只复用只读指标和报告能力，不接券商交易模块。

`ashare-live-research` 是独立的本机实时观察 skill/MCP 项目，负责同花顺、手机镜像、行情异动和辅助交易研究的只读观察层。它不并入 DSA，不读取密码，不自动下单；DSA 只把它当可发现 skill 和后续只读联动能力。

`a-share-us-catalyst` / CatDesk（九点猫研）是独立的美股催化 A 股题材观察服务。DSA 当前只提供本地跳转和服务状态提示；它的“候选信号历史验证”不等同于 Qlib 模型/策略回测，不应把术语混写。维护口径是：九点猫研负责发现题材和候选股；DSA、AlphaSift 和 Qlib 负责进一步过滤、验证、回测和风险评估。

`quant_data`、Qlib provider、MLflow `mlruns`、下载缓存、日志和报告是本地数据资产。它们不应提交到 Git。

JQData 凭据的首选位置是 `~/.agents/private/quant-workbench/jqdata.env`，也可通过 `JQDATA_ENV_FILE` 指向其他本机私有文件。不要把 `JQ_USERNAME` / `JQ_PASSWORD` 写入 DSA 仓库、Qlib 源码目录或交接报告。历史兼容路径 `WORKING_PROJECTS_ROOT/qlib/.env` 只能作为迁移期 fallback；发现该文件含真实凭据时，应先迁移到私有目录并确认脚本仍可读取。

## Agent 开工检查

每个 agent 接手前先执行或确认：

```bash
pwd
git status --short
```

然后读取：

```bash
sed -n '1,220p' AGENTS.md
sed -n '1,220p' docs/agent-maintenance-contract.md
sed -n '1,220p' docs/local-quant-workbench.md
```

如果任务涉及安装、服务启动或新用户复现，还要读取：

```bash
sed -n '1,260p' docs/install-local-quant-workbench.md
```

如果任务涉及前端页面，检查 `apps/dsa-web/` 的现有测试和路由。
如果任务涉及后端任务中心，检查 `src/services/quant_job_service.py`、`api/v1/endpoints/quant_jobs.py` 和 `tests/test_quant_jobs_api.py`。
如果任务涉及数据治理或报告，检查 `docs/local-quant-workbench.md` 和对应 `quant_workflow/scripts` 脚本。

## 修改原则

1. 不执行 `git commit`、`git push`、`git tag`，除非用户明确要求。
2. 不删除、移动、归档真实数据目录，除非用户明确确认具体清单。
3. 不打印、复制、提交 `.env`、token、cookie、JQData 密码、LLM key 或本机私密路径。
4. 不让网页任务执行任意命令。新增任务必须通过后端白名单映射到固定脚本和固定参数。
5. 不把全量数据、`mlruns`、日志、`.venv`、`.codegraph`、`reports`、`data` 提交到 Git。
6. 不把 DSA、Qlib、MLflow 混成一个概念。文档和 UI 必须分别说明三者职责。
7. 不将 ETF/场内基金误按单家公司财报逻辑分析。ETF 基本面必须标注降级或待补强。
8. 不接券商自动交易。本阶段同花顺和手机镜像只作为后续只读观察方向。

## 用户沟通口径

所有回答都必须避免模棱两可和误导。agent 必须优先使用通俗、明确、可核对的话说明自己做了什么、没做什么、依据是什么、结论边界是什么。不要用听起来专业但用户无法判断真假的笼统说法。

涉及数据日期、样例/实时、训练/推理、预测/回测、候选股/买入信号、报告生成时间/数据覆盖时间时，这条规则尤其严格。不要只说“更新了”“跑预测”“数据可用”。应拆开说明：读取了哪个文件、用了哪一天之前的数据、没有使用哪一天的数据、生成了什么报告、报告能说明什么、不能说明什么。若发现旧说法可能误导，应主动更正。

## 可追溯要求

每次改动必须能回答五个问题：

| 问题 | 记录位置 |
| --- | --- |
| 为什么改 | 用户需求、issue、任务说明或交接报告 |
| 改了什么 | `git diff`、最终交付说明 |
| 哪些文件受影响 | 最终交付说明、必要时 `docs/CHANGELOG.md` |
| 怎么验证 | 测试命令和结果 |
| 怎么回滚 | 最终交付说明或文档中的回滚段落 |

用户可见能力、API 语义、安装方式、安全边界、数据路径、实验删除语义或工作流任务变化，必须更新 `docs/CHANGELOG.md` 的 `[Unreleased]` 段。该段使用扁平格式：

```markdown
- [文档] 描述
- [改进] 描述
- [修复] 描述
```

不要在 `[Unreleased]` 里新增二级标题。

## Agent 交接模板

当把工作交给另一个 agent 时，使用下面模板。不要只写“继续优化”。

```markdown
# Agent Handoff

## Objective
一句话说明本轮目标。

## Workspace
- Repo: `/path/to/daily_stock_analysis`
- Embedded workflow: `quant_workflow/`
- Optional related repo: `/path/to/china-astock-quant`
- Local UI: `http://127.0.0.1:8010/`
- MLflow: `http://127.0.0.1:6006/`

## Required Reading
- `AGENTS.md`
- `docs/agent-maintenance-contract.md`
- `docs/local-quant-workbench.md`
- 任务相关源码或测试文件

## Current State
- `git status --short` 摘要
- 已启动服务
- 已完成验证

## Scope
- 允许修改：
- 禁止修改：

## Safety Rules
- 不提交、不推送，除非用户明确要求
- 不删除真实数据
- 不泄露 `.env` 或 token
- 量化任务只能走白名单脚本

## Expected Deliverable
- 需要交付的代码、文档、测试或报告

## Verification
- 必跑命令：
- 可选命令：

## Open Risks
- 已知风险或未验证项
```

## 验证分级

文档或索引改动：

```bash
python scripts/check_ai_assets.py
```

如果只修改 `docs/**` 且不碰 `AGENTS.md`、`.github` 指令或 `.claude/skills`，可不跑代码测试，但最终说明要写明“文档-only，未跑后端/前端测试”。

量化工作台 focused 改动：

```bash
scripts/verify-local-quant-workbench.sh quick
```

阶段验收：

```bash
scripts/verify-local-quant-workbench.sh full
```

最终闭环：

```bash
scripts/verify-local-quant-workbench.sh complete
```

`complete` 会在 `full` 后继续跑 opencode 只读 review。若 opencode 报 P0/P1，必须先核实并修复真实问题，再重新跑相关验证。

## 服务启动约定

本机默认入口：

| 服务 | 地址 | 说明 |
| --- | --- | --- |
| DSA | `http://127.0.0.1:8010/` | 主网页和 API |
| DSA 实验入口 | `http://127.0.0.1:8010/experiments/` | MLflow 官方 UI 的双语用户入口 |
| MLflow 后台 | `http://127.0.0.1:6006/` | 仅提供实验数据和官方静态资源，不作为日常入口 |

启动 MLflow 时保留本地 file store 显式开关：

```bash
MLFLOW_STORE="${QUANT_WORKFLOW_MLRUNS_DIR:-/path/to/daily_stock_analysis/quant_workflow/mlruns}"
MLFLOW_ALLOW_FILE_STORE=true .venv/bin/python -m mlflow ui \
  --backend-store-uri "file://$MLFLOW_STORE" \
  --host 127.0.0.1 \
  --port 6006
```

长跑服务或下载任务应使用 `screen`、`tmux`、`nohup`、`launchd` 或项目持久任务机制，并写入日志。不要把正式长任务裸跑在一次性前台 shell 里。

MLflow 双语层维护规则：不要 clone 或修改 `.venv` 中的 MLflow 源码；不要启动第二个 MLflow 端口；修改 `src/services/mlflow_ui_proxy.py` 或翻译目录后，至少运行 `tests/test_mlflow_ui_proxy.py`、`tests/test_mlflow_service.py`、目录生成器测试、前端链接测试和 `e2e/verify-mlflow-bilingual.spec.ts`。当前 MLflow `3.14.0` 对应上游 commit `86cd7f5a1dc25a1387ad07c87811bbc30f62951c`。升级时必须同时更新版本、上游 commit、翻译源 URL 和生成产物，并在浏览器核对实验列表、Run 详情、筛选和 artifact；生成器的版本检查不得绕过。

如果 agent 或外部脚本绕过 `main.py` / `webui.py` / `server.py`，直接用 `uvicorn.Config(host=...)` 创建服务，必须先设置 `DSA_BIND_HOST` 或调用 `src.web_bind.publish_bind_host(host)`。否则应用工厂无法看到真实 host，公网绑定认证检查只能按 localhost 默认处理。

## GitHub 分发边界

可以提交和推送：

- DSA 源码
- 仓内 `quant_workflow/` 的脚本、配置、测试和文档
- 前后端测试
- 文档
- `.env.example`
- 可复现脚本和白名单任务定义

不应提交或推送：

- `.env`
- `.venv`
- `data/`
- `reports/`
- `logs/`
- Qlib provider 数据
- `quant_data`
- MLflow `mlruns`
- JQData、Tushare、LLM、券商或浏览器凭据

其他用户安装时应 clone DSA 仓库，按 `docs/install-local-quant-workbench.md` 安装依赖，通过 PyPI 安装 Qlib 和 MLflow，按本地路径准备自己的数据。不要要求普通用户 fork 或修改官方 Qlib 仓库。
