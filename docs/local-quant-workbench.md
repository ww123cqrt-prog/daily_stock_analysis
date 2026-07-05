# 本地统一量化研究与回测工作台核对手册

本文用于核对 `daily_stock_analysis`、`quant_workflow`、`Qlib`、`china-astock-quant` 和本机只读参考工具之间的协作关系。目标是让你主要通过网页完成日常研究、数据检查、Qlib 转换、训练、回测、实验查看和历史管理，而不是依赖终端手工命令。

## 一句话架构

`daily_stock_analysis` 是主网页和主操作台；`quant_workflow` 是本地量化流水线；`Qlib` 是机器学习量化引擎；`china-astock-quant` 是 A 股指标、策略和回测经验参考库；MLflow 只保留为实验原始记录页面。唯一 Python 虚拟环境是 `daily_stock_analysis/.venv`：DSA 后端、`quant_workflow` 稳定脚本、Qlib、JQData/AkShare 和 `china-astock-quant` 适配层都从这套环境运行。

## 本机入口

| 入口 | 地址 | 用途 |
| --- | --- | --- |
| DSA 主网页 | `http://127.0.0.1:8010/` | 股票分析、研究、工作流、回测、问股、持仓、告警、选股 |
| 工作流任务中心 | `/workflow` | 从网页运行白名单量化脚本、查看状态、日志和产物 |
| 研究总览 | `/research` | 查看数据状态、项目关系、实验记录、能力边界和复用清单 |
| 回测页 | `/backtest` | 查看 DSA 历史准确性回测和 Qlib 模型/策略回测摘要 |
| MLflow | `http://127.0.0.1:6006/` | 查看实验原始 run、参数和 artifact，不作为主 UI |

如果 `8000` 被其他服务占用，以当前实际启动的 `8010` 为准；页面文案不应假定 `8000` 就是 DSA。

## 项目职责

| 项目 | 当前角色 | 应该放什么 | 不应该做什么 |
| --- | --- | --- | --- |
| `daily_stock_analysis` | 主网页、API、用户操作台 | 页面、后端 API、任务中心、实验摘要、回测展示 | 不保存真实密码/token，不直接接券商交易 |
| `quant_workflow` | 稳定脚本和产物层 | 数据下载、清洗、异常报告、Qlib 转换、训练、回测、报告 | 不把一次性脚本散落到项目根目录 |
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
5. 质量报告：运行“生成数据质量报告”和“生成异常股票和清洗策略”。
6. 清洗预演：先用“只预演不写入”，确认影响范围后再考虑生成 cleaned provider。
7. 训练+回测：运行“训练 Qlib LightGBM 基线”，再运行“生成 Qlib 预测摘要”、市场状态、现金仓位叠加、策略优化预演、DSA 分析准确性回测。
8. 复用参考：运行“生成 china-astock 策略对比”，只生成报告，不连接券商。

每个任务只能调用后端白名单脚本；网页不能提交任意 shell 命令。

### 2.1 quant_workflow 脚本覆盖盘点

`quant_workflow/scripts` 里的稳定脚本不等于全部都应该直接放到网页按钮里。当前网页任务中心已经覆盖核心日常链路：数据下载、JQData dry-run、Qlib 合并、清洗、检查、质量报告、异常报告、市场状态、现金仓位叠加、china-astock 对比、策略优化预演和 baseline 训练。

以下脚本暂时保留为脚本层能力：

- `add_index_benchmark.py`：provider 维护脚本，会补写沪深300 benchmark 特征；进入网页前需要先补 dry-run、参数化 provider 路径和产物预览。
- `generate_csi300.py`：provider 维护脚本，会生成 `csi300.txt` 股票池；进入网页前需要先补 dry-run 和目标 provider 参数。
- `supervise_baostock.py`：长下载监督器，和网页任务中心自己的持久任务、日志和取消能力重叠；暂不单独暴露，优先用网页里的 Baostock 下载任务和任务日志。

这三项会在 `/research` 的“功能入口盘点”中显示，避免误以为脚本被遗漏。

### 3. 查看回测和实验

打开 `http://127.0.0.1:8010/backtest`。

- `DSA分析准确性回测`：看历史分析建议和后续走势的一致性。
- `Qlib模型/策略回测`：看最新训练摘要、关键指标、Qlib 预测摘要、cash-control 对比、MLflow 链接和产物位置。
- `Qlib 预测摘要`：读取最新 MLflow `pred.pkl`，展示最新交易日的最高/最低模型评分，并写入 `qlib_predictions_latest.{json,md}`；它是只读排行，不是自动买卖信号。
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
| 稳定量化脚本 | `quant_workflow/scripts/` |
| 量化报告和实验记录 | `quant_workflow/reports/` |
| 临时实验草稿 | `quant_workflow/scratch/` |
| Qlib provider 和行情数据 | `quant_data/` |
| CodeGraph 索引 | 每个项目自己的 `.codegraph/` |
| Python 虚拟环境 | `daily_stock_analysis/.venv/` 是唯一 Python 运行时；`qlib/.venv` 和 `china-astock-quant/.venv` 不作为网页任务默认运行环境 |

可迁移路径统一通过环境变量覆盖：`QUANT_DATA_ROOT` 指向行情数据和 Qlib provider 根目录，`MLFLOW_URL` 控制网页里展示的 MLflow 链接。默认值分别是 `/Users/cq/WorkingProjects/quant_data` 和 `http://127.0.0.1:6006/`。

不要把缓存、日志、报告、数据、`.venv/`、`.codegraph/`、`.pytest_cache/`、`__pycache__/` 提交到 Git。

## 自动验证命令

在 `daily_stock_analysis` 中：

```bash
.venv/bin/python -m pytest tests/test_quant_jobs_api.py tests/test_research_status_api.py tests/test_quant_workflow_docs.py -q
cd apps/dsa-web
npm test -- --run src/pages/__tests__/WorkflowPage.test.tsx src/pages/__tests__/ResearchPage.test.tsx src/pages/__tests__/BacktestPage.test.tsx src/api/__tests__/quant.test.ts src/api/__tests__/research.test.ts
npm run lint
npm run build
npm run test:smoke:fake-analysis
npm run test:smoke:local
```

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
