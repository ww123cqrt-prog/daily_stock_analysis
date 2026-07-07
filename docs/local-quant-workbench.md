# 本地统一量化研究与回测工作台核对手册

本文用于核对 `daily_stock_analysis`、`quant_workflow`、`Qlib`、`china-astock-quant` 和本机只读参考工具之间的协作关系。目标是让你主要通过网页完成日常研究、数据检查、Qlib 转换、训练、回测、实验查看和历史管理，而不是依赖终端手工命令。

## 一句话架构

`daily_stock_analysis` 是主网页和主操作台；仓内 `quant_workflow/` 是本地量化流水线；`Qlib` 是机器学习量化引擎；`china-astock-quant` 是 A 股指标、策略和回测经验参考库；MLflow 只保留为实验原始记录页面。唯一 Python 虚拟环境是 `daily_stock_analysis/.venv`：DSA 后端、`quant_workflow` 稳定脚本、Qlib、JQData/AkShare 和 `china-astock-quant` 适配层都从这套环境运行。

## 本机入口

| 入口 | 地址 | 用途 |
| --- | --- | --- |
| DSA 主网页 | `http://127.0.0.1:8010/` | 股票分析、研究、工作流、回测、问股、持仓、告警、选股 |
| 工作流任务中心 | `/workflow` | 从网页运行白名单量化脚本、查看状态、日志和产物 |
| 研究总览 | `/research` | 查看数据状态、项目关系、实验记录、能力边界和复用清单 |
| 回测页 | `/backtest` | 查看 DSA 历史准确性回测和 Qlib 模型/策略回测摘要 |
| MLflow | `http://127.0.0.1:6006/` | 查看实验原始 run、参数和 artifact，不作为主 UI |

如果 `8000` 被其他服务占用，以当前实际启动的 `8010` 为准；页面文案不应假定 `8000` 就是 DSA。

## 本地安全边界

本地研究默认只绑定 `127.0.0.1`。只要 `DSA_BIND_HOST`/`UVICORN_HOST` 指向非 localhost，或 `DSA_ENV=production`，后端就要求 `ADMIN_AUTH_ENABLED=true`；否则启动会直接拒绝。设置页也不能在这些暴露运行时关闭认证。需要局域网或公网访问时，先启用管理员认证，再配置明确的来源列表或反向代理。

`/api/v1/quant/*` 的任务创建/取消/SSE，以及 `/api/v1/research/experiments/*` 的实验记录变更，都要求请求带有本机 `Origin` 或本机 `Referer`。`/api/v1/quant/*` 的任务列表、日志、产物和状态读取，以及 `/api/v1/research/status` 也会拒绝跨站浏览器读取；无 `Origin` 的本机 CLI 健康检查仍可读取。浏览器里的 DSA 页面会自动满足这个条件；裸 `curl`/脚本调试 mutation API 时，应显式带上本机 `Referer: http://127.0.0.1:8010/workflow` 或 `Origin: http://127.0.0.1:8010`。`X-DSA-Local-Client` 不作为来源保护旁路；一旦绑定到 `0.0.0.0` 或生产环境，应启用管理员认证后按受控客户端方式接入。

量化任务接口在本地模式下可以不启用登录认证，但安全边界是“只绑定 localhost + 来源校验 + 白名单脚本”。如果要给局域网或公网使用，必须启用 `ADMIN_AUTH_ENABLED=true`，并额外评估访问控制、反向代理和提交频率限制。

量化任务中心启动脚本时只向子进程传递白名单环境变量：`CHILD_ENV_EXACT_NAMES` 和 `CHILD_ENV_PREFIXES` 默认只包含路径、运行时和非敏感数据配置；`JQ_USERNAME`、`JQ_PASSWORD`、`TUSHARE_TOKEN`、`AKSHARE_TOKEN` 这类数据源凭据不会进入所有任务。只有任务 spec 显式声明 `env_names` / `env_prefixes` 时，对应 job 才能拿到必要凭据，例如 `download_jqdata` 只允许 JQData 登录变量。LLM/API provider key、`MLFLOW_URL`、`MLFLOW_*`、未列入白名单的 `QUANT_API_KEY` / `QUANT_SECRET_TOKEN` 和其他系统环境不会原样继承给 durable runner。新增任务脚本时也要遵守同一边界：不要把 token 放进命令参数，不要打印 `os.environ`，不要把密码写入日志或报告。

`MLFLOW_URL` 只用于网页链接和本地 health check。health check 只探测 `127.0.0.1`、`localhost` 或 `::1`，并拒绝带用户名密码或路径穿越段的 URL；API 返回给前端的展示 URL 会移除 username/password、query 和 fragment。非本机 MLflow 地址只展示为外部链接，不主动探测。`MAX_QUANT_JOB_STREAMS` 可调节 `/workflow` SSE 连接上限，默认 8，最小值按 1 处理。

任务状态 API 不返回真实 `cwd`、`job.log` 路径或 artifact 绝对路径；前端只拿 artifact 展示名和索引，下载/预览时仍由后端按 job id 与 artifact index 做路径边界检查。研究状态里的数据治理路径也只返回项目内标签或文件名，`.maintenance.lock` 只暴露锁定状态和更新时间，不暴露文件内容。

工作流页面的实时任务列表使用 SSE 长连接。直连本机通常会在浏览器关闭页面时释放连接；如果未来通过 Nginx、内网穿透或其他反向代理暴露页面，应配置合理的读超时和空闲连接回收，避免异常断线长期占用 SSE 连接槽。多 worker 部署下每个 worker 都有自己的 SSE 连接计数，因此有效上限约为 `worker 数量 × MAX_QUANT_JOB_STREAMS`；当前工作台按单机本地服务设计。

实验记录本的归档、永久删除和训练脚本追加记录共用 `quant_workflow/reports/.experiment.lock`，用于避免网页删除记录和后台训练同时写 `experiment_runs.jsonl` / `experiment_log.md` 时互相覆盖。这个锁是本机协作式 advisory lock：只有走 DSA API 或按同一约定实现的 `quant_workflow` 写入流程才会遵守它；手工编辑、`echo >>` 或第三方脚本直接写文件仍可绕过锁，因此不要在训练或网页删除实验时手工改这两个文件。

## 项目职责

| 项目 | 当前角色 | 应该放什么 | 不应该做什么 |
| --- | --- | --- | --- |
| `daily_stock_analysis` | 主网页、API、用户操作台 | 页面、后端 API、任务中心、实验摘要、回测展示 | 不保存真实密码/token，不直接接券商交易 |
| `quant_workflow` | DSA 仓内稳定脚本和产物层 | 数据下载、清洗、异常报告、Qlib 转换、训练、回测、报告 | 不把一次性脚本散落到项目根目录，不提交运行产物 |
| `Qlib` | 机器学习量化引擎 | 由 `daily_stock_analysis/.venv` 中的 PyPI 包调用，读取本地 provider，执行模型训练和回测 | 不改造成主网页，不随意改官方源码 |
| `china-astock-quant` | 参考库和复用来源 | 由共享 DSA Python 环境调用其指标、样例策略、事件回测和工程经验 | 不接入交易模块，不改造成主网页 |
| 同花顺桌面端 | 后续只读观察方向 | 界面参考、人工核对、未来只读截图/OCR 研究 | 不读取账号密码，不自动下单，不连接券商交易接口 |

## 网页自助流程

### 1. 检查当前状态

打开 `http://127.0.0.1:8010/research`。

确认以下区域可见：

- 数据状态：是否可训练、数据源、Qlib provider 日期范围。
- 实验记录本：最近训练、关键指标、MLflow 链接。
- 能力验收矩阵：DSA、quant_workflow、Qlib、china-astock-quant、同花顺、交易执行禁区。
- 功能入口盘点：每个入口的状态、缺口和下一步。
- `china-astock-quant` 复用清单：指标、策略、事件回测、纸面交易禁区。

### 2. 运行量化流水线

打开 `http://127.0.0.1:8010/workflow`。

建议顺序：

1. 检查 Qlib 数据：运行“检查 Qlib 数据”。
2. 更新数据：先运行“准备 Baostock 增量清单”，再按需运行“下载 Baostock 长历史”。
3. 验证 JQData：运行“验证 JQData 连接”。网页端只允许 dry-run、小样本认证和股票池解析，不写入真实数据目录。
4. 合并 Qlib：运行“合并并转换为 Qlib”，必要时先用小样本演练。
5. 质量报告：运行“生成数据质量报告”、“生成异常股票和清洗策略”、“生成数据治理盘点和清理预案”和“生成工作台能力盘点”。
6. 清洗预演：先用“只预演不写入”，确认影响范围后再考虑生成 cleaned provider。
7. 训练+回测：运行“训练 Qlib LightGBM 基线”，再运行“生成 Qlib 预测摘要”、市场状态、现金仓位叠加、策略优化预演、DSA 分析准确性回测。
8. Qlib 研究增强：运行“生成 Qlib 模型解释报告”、“生成 Qlib 回测明细报告”、“生成 Qlib 多模型/多实验对比”、“生成 Qlib 因子分析报告”、“生成 Qlib 滚动训练计划”和“生成 Qlib 风险模型诊断”。
9. 复用参考：运行“生成 china-astock 技术指标报告”和“生成 china-astock 策略对比”，只生成报告，不连接券商。

每个任务只能调用后端白名单脚本；网页不能提交任意 shell 命令。

### 2.1 quant_workflow 脚本覆盖盘点

`quant_workflow/scripts` 里的稳定脚本不等于全部都应该直接放到网页按钮里。当前网页任务中心已经覆盖核心日常链路：数据下载、JQData dry-run、Qlib 合并、清洗、检查、质量报告、异常报告、数据治理盘点和 dry-run 清理预案、市场状态、现金仓位叠加、Qlib 研究增强、china-astock 技术指标报告、china-astock 对比、策略优化预演和 baseline 训练。

`report_data_hygiene.py` 在 `/workflow` 中作为“生成数据治理盘点和清理预案”任务出现。它只读检查 `quant_data`、`quant_workflow`、DSA、Qlib、china-astock-quant 的当前体量、任务库状态、业务进程、官方 parquet、官方 Qlib provider、根目录日志/临时脚本和历史报告，输出 `data_hygiene_inventory_latest.{md,json}` 与 `data_hygiene_cleanup_plan_latest.{md,json}`。清理计划只包含 `keep`、`document`、`move`、`archive` 等 dry-run 建议，不包含真实删除动作；所有会改变路径的建议都标记为需要用户确认。

`report_workbench_capabilities.py` 在 `/workflow` 中作为“生成工作台能力盘点”任务出现。它只读扫描 DSA 前端页面、DSA API router、后端量化任务白名单、网页任务卡、`quant_workflow/scripts` 稳定脚本、测试覆盖和 `china-astock-quant` 可复用边界，输出 `workbench_capability_inventory_latest.{json,md}`。这个任务用于回答“哪些能力已经有网页入口，哪些还缺测试、缺前端入口或只保留在脚本层”，不启动下载、转换、清洗、训练任务，也不接券商、不自动下单。

`report_china_astock_indicators.py` 复用 `china-astock-quant/strategies/_indicators.py` 的 SMA、EMA、RSI、ATR 和 Bollinger 指标函数，读取本地 cleaned 行情生成 `china_astock_indicator_report_latest.{json,md}`。这是只读研究报告，不连接券商、不读取账号密码、不自动下单。网页端和脚本端省略 `limit` 时都默认只抽样 50 只股票，避免误把“安全报告”跑成全市场重任务；需要扩大范围时必须显式填写数量上限。

`report_qlib_research.py` 在 `/workflow` 中作为“Qlib 研究增强”的 6 个固定任务出现：模型效果深度解释、回测详细分解、多模型/多实验对比、因子挖掘辅助、滚动训练计划和风险模型集成诊断。每个网页按钮都固定传入对应 section，不能从网页提交任意 section 或 shell 命令。脚本读取本地 MLflow/Qlib 既有产物生成 `qlib_research_model_explainability_latest.{json,md}` 等报告；首版不重新训练、不计算无限制 SHAP、不接券商、不自动下单。

研究状态 API 只会对配置的 `quant_data/qlib` provider 根目录内的 Qlib 路径补充日历、标的和体量信息。`latest_training_result.json` 如果指向根目录之外的路径，页面会保留警告，不递归扫描该目录。

以下 provider 维护脚本已经进入网页，但只允许安全预演：

- `add_index_benchmark.py`：在 `/workflow` 中作为“预演添加沪深300 benchmark”任务出现，后端强制 `--dry-run`，不请求 Baostock、不写 bin、不改股票池。
- `generate_csi300.py`：在 `/workflow` 中作为“预演生成 CSI300 股票池”任务出现，后端强制 `--dry-run`，只读取缓存股票池和 Qlib 日历并打印预演结果；缓存缺失时直接失败，不走 AkShare fallback。
- `repair_qlib_provider.py`：在 `/workflow` 中作为“预演修复 Qlib Provider”任务出现，后端强制 `--dry-run`，兼容旧客户端传入 `dry_run=true`，但拒绝 `dry_run=false`；只输出修复计划，不重命名、不写入、不删除 provider 文件。

以下脚本暂时保留为脚本层能力：

- `supervise_baostock.py`：长下载监督器，和网页任务中心自己的持久任务、日志和取消能力重叠；暂不单独暴露，优先用网页里的 Baostock 下载任务和任务日志。

这三项会在 `/research` 的“功能入口盘点”中显示，避免误以为脚本被遗漏。

### 3. 查看回测和实验

打开 `http://127.0.0.1:8010/backtest`。

- `DSA分析准确性回测`：看历史分析建议和后续走势的一致性。
- `Qlib模型/策略回测`：看最新训练摘要、关键指标、Qlib 预测摘要、cash-control 对比、MLflow 链接和产物位置。
- `Qlib 预测摘要`：读取最新 MLflow `pred.pkl`，展示最新交易日的最高/最低模型评分，并写入 `qlib_predictions_latest.{json,md}`；它是只读排行，不是自动买卖信号。
- `Qlib 研究增强`：读取最新或指定 MLflow run 的 `pred.pkl`、`label.pkl`、`params.pkl` 和 `portfolio_analysis`，生成模型解释、回测明细、多实验对比、因子 IC、滚动训练计划和风险诊断报告。
- `实验复现与产物`：在 Qlib 标签页查看本次训练使用的数据版本、provider 路径、source config、runtime config、MLflow run id 和关键 artifact，例如 `pred.pkl`、`portfolio_analysis`、`task`。
- `策略优化预演`：在 `/workflow` 用内置小样本 dry-run 验证低换手、市场择时和回撤保护实验链路；它只读生成报告，不训练、不连接券商、不自动下单。

打开 `http://127.0.0.1:8010/research` 查看实验记录本：

- “归档”会从当前活跃实验列表移除，并写入归档 JSONL。
- “永久删除”会真实删除 `experiment_runs.jsonl` 中对应记录，并删除带有 `<!-- experiment-run:... -->` 标记的 Markdown 实验块。
- 永久删除需要二次确认；未标记或不完整的 Markdown 块不会被静默删除。

## 场内基金/ETF 边界

场内基金/ETF 可以复用趋势、均线、MACD、RSI/KDJ、成交量、流动性、主题/指数逻辑、新闻和政策影响分析。

ETF 的基本面分析会降级：不要把股票公司的财报、ROE、公司竞争力、高管风险直接套到 ETF 上。后续 ETF 深度版再补折溢价、跟踪误差、规模份额、持仓、基金类型和同类 ETF 对比。

## 安全边界

必须长期遵守：

- 不自动下单。
- 不读取账号密码。
- 不连接券商交易接口。
- 不打印、不提交、不记录真实 token、cookie、JQData 密码或 LLM key。
- `.env` 只本地使用；示例文件只写占位符。
- 同花顺和手机镜像只作为后续只读观察方向，手机连接本阶段不做。

## 文件和产物规则

| 类型 | 位置 |
| --- | --- |
| DSA 代码 | `daily_stock_analysis/` |
| 稳定量化脚本 | `daily_stock_analysis/quant_workflow/scripts/` |
| 量化报告和实验记录 | `daily_stock_analysis/quant_workflow/reports/`，运行态产物，不入库 |
| 临时实验草稿 | `daily_stock_analysis/quant_workflow/scratch/`，只保留 README |
| Qlib provider 和行情数据 | `daily_stock_analysis/quant_data/`，运行态数据，不入库 |
| CodeGraph 索引 | 每个项目自己的 `.codegraph/` |
| Python 虚拟环境 | `daily_stock_analysis/.venv/` 是唯一 Python 运行时；`qlib/.venv` 和 `china-astock-quant/.venv` 不作为网页任务默认运行环境 |

可迁移路径统一通过环境变量覆盖：`QUANT_DATA_ROOT` 指向行情数据和 Qlib provider 根目录，`QUANT_WORKFLOW_ROOT` 指向量化脚本层，`MLFLOW_URL` 控制网页里展示的 MLflow 链接，`MAX_QUANT_JOB_STREAMS` 控制工作流实时状态连接上限。默认值分别是仓内 `quant_data/`、仓内 `quant_workflow/`、`http://127.0.0.1:6006/` 和 `8`。

量化任务统一使用 `QUANT_WORKFLOW_PYTHON`，默认指向 `daily_stock_analysis/.venv/bin/python`。旧的 `QLIB_PYTHON` 不再作为网页任务中心的 Python 入口；如果其他脚本仍设置了 `QLIB_PYTHON`，不会覆盖 DSA 网页任务运行环境。

新增 `quant_workflow` 网页任务时，如果脚本依赖新的环境变量，必须同步评估并更新 `src/services/quant_job_service.py` 里的默认非敏感白名单，或在该 job spec 上显式声明 `env_names` / `env_prefixes`。任务中心不会把 DSA API 进程的完整环境直接透传给子进程；漏配白名单会导致脚本在网页端缺少变量，误把所有变量放开则会增加 token、密码或 LLM key 暴露到日志的风险。

`download_jqdata` 在网页端固定为 dry-run，是“验证 JQData 登录和连接是否可用”，不是全量下载。该任务只向子进程透传 `JQ_USERNAME` 和 `JQ_PASSWORD`；日志只记录命令和状态，不应打印或持久化真实密码。

不要把缓存、日志、报告、数据、`.venv/`、`.codegraph/`、`.pytest_cache/`、`__pycache__/` 提交到 Git。

## 自动验证命令

推荐先在 `daily_stock_analysis` 中使用统一验收脚本：

```bash
scripts/verify-local-quant-workbench.sh quick
scripts/verify-local-quant-workbench.sh full
scripts/verify-local-quant-workbench.sh opencode-review
scripts/verify-local-quant-workbench.sh complete
```

`quick` 运行本轮工作台相关的后端/前端 focused 测试；`full` 依次运行 DSA 后端全量、前端 test/lint/build、`quant_workflow` 脚本测试、数据治理浏览器 smoke、托管 fake-analysis 浏览器 smoke 和本机 8010 浏览器 smoke；`opencode-review` 用只读方式审查当前工作树，包括 `git diff HEAD`、`git status --short` 和未跟踪正式文件；`complete` 表示 `full` 后继续执行 `opencode-review`，用于最终阶段闭环。

也可以手动拆开运行：

```bash
.venv/bin/python -m pytest tests/test_api_app_cors.py tests/test_auth.py tests/test_web_bind.py tests/test_main_schedule_mode.py tests/test_quant_jobs_api.py tests/test_research_status_api.py tests/test_quant_workflow_docs.py tests/test_stocks_api.py -q
cd apps/dsa-web
npm test -- --run src/pages/__tests__/WorkflowPage.test.tsx src/pages/__tests__/ResearchPage.test.tsx src/pages/__tests__/BacktestPage.test.tsx src/api/__tests__/quant.test.ts src/api/__tests__/research.test.ts
npm run lint
npm run build
npm run test:smoke:fake-analysis
npm run test:smoke:local
```

如果只想快速核对量化任务 API，可单独运行 `pytest tests/test_quant_jobs_api.py -q`。

`test:smoke:fake-analysis` 会托管临时后端和前端，并把量化 smoke 产物写到
`apps/dsa-web/test-results/managed-smoke/` 下的隔离 reports，不覆盖真实
`quant_workflow/reports/*_latest` 指针。`test:smoke:local` 面向当前正在运行的
本机 8010 服务，适合最终验收前手动确认。

在 `quant_workflow` 中：

```bash
python -m pytest tests -q
```

浏览器验收必须至少打开：

- `http://127.0.0.1:8010/research`
- `http://127.0.0.1:8010/workflow`
- `http://127.0.0.1:8010/backtest`

并确认页面可渲染、按钮可点击、API 可调用、错误状态可显示、产物入口可访问。

## 常见异常判断

| 现象 | 优先看哪里 |
| --- | --- |
| 页面打不开 | DSA 服务是否在 `8010`，端口是否被占用 |
| 工作流按钮失败 | `/workflow` 任务日志抽屉、`quant_workflow/reports/jobs/<job_id>/job.log` |
| Qlib 不可训练 | `/research` 的数据状态和 `check_qlib_data` 任务日志 |
| 数据质量差 | 最新数据质量报告、异常股票清单、清洗策略报告 |
| 训练效果差 | `/backtest` 的 Qlib 指标、MLflow、cash-control 对比 |
| 实验删不掉 | 确认 run id 存在，Markdown 实验块是否带完整标记 |

## 阶段验收清单

- DSA 工作区无散落未跟踪源码。
- `/research`、`/workflow`、`/backtest` 能在浏览器打开。
- 工作流白名单任务均有网页入口；策略优化预演默认 dry-run，只作为安全链路验证。
- Qlib 数据检查、转换、训练、回测摘要可以从网页触发或查看。
- MLflow 链接存在，但不替代 DSA 主 UI。
- 实验记录支持归档和永久删除。
- `china-astock-quant` 的复用能力已列账，交易相关能力保持禁区。
- 同花顺只作为只读观察方向，手机连接暂不做。
- 后端 pytest、前端 test/lint/build、浏览器 smoke、opencode 只读 review 都完成后，才算阶段闭环。
