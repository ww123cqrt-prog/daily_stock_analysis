# 本地统一量化研究与回测工作台核对手册

本文用于核对 `daily_stock_analysis`、`quant_data`、`quant_workflow`、`Qlib`、`china-astock-quant` 和本机只读参考工具之间的协作关系。DSA 网页只保留日常入口、只读数据状态、量化研究、个股分析、持仓和设置；运行管理与历史兼容路由继续由后端保留，但不再作为一级用户功能。

## 一句话架构

当前框架是：入口统一到 DSA，数据治理留在 `quant_data`，研究执行留在仓内 `quant_workflow/` 与 Qlib。DSA 通过本机数据服务读取不可变快照身份和可用状态，不下载、不清洗、不发布数据。`china-astock-quant` 是 A 股指标、策略和回测经验参考库；`ashare-live-research` 是独立只读实时观察 skill；`a-share-us-catalyst` / CatDesk 是独立催化观察服务；MLflow 保存实验原始记录，页面通过 DSA `/experiments/` 访问。

## 默认 A 股股票费用口径

本项目用户指定的 A 股股票回测与交易记录默认费用口径如下；除非某个策略或券商配置明确声明其他口径，不得自行替换为平台默认费率：

- 佣金：万 0.73，即 `0.000073`，买卖双边收取。
- 最低佣金：0 元（免五）。
- 印花税：卖出千一，即 `0.001`；买入不收取。
- 过户费：买卖双边十万分之一，即 `0.00001`。

报告必须明确记录实际使用的费率、最低佣金、印花税和过户费。ETF 策略是否收取印花税或过户费，应按该策略自身的资产类型和费用定义单独说明，不能套用股票口径。

## 本机入口

| 入口 | 地址 | 用途 |
| --- | --- | --- |
| 首页 | `/` | 跳转到 DSA 功能、九点猫研和 MLflow，并以静态关系图说明各项目职责与数据流 |
| 数据 | `/data` | 只读查看 `quant_data` 服务、正式数据集和不可变快照身份 |
| 量化研究 | `/research` | 在“选股研究”和“交易策略研究”之间切换 |
| 个股分析 | `/stock-analysis` | 运行个股分析并查看历史报告 |
| 持仓 | `/portfolio` | 管理账户、持仓、估值、风险和手工交易记录 |
| 设置 | `/settings` | 管理 DSA 本地配置 |
| MLflow | `/experiments/` | 查看实验、Run、参数、指标和 artifact |

`/workflow`、`/backtest`、`/other` 等旧地址继续兼容已有书签和测试，但不进入一级导航。实际服务端口以启动记录和监听进程为准，页面文案不得永久假定某个端口就是 DSA。

数据页不直接访问文件系统。DSA 的 `GET /api/v1/data/status` 只读代理 `quant_data` 数据服务（默认 `http://127.0.0.1:8766`），服务异常时返回 503；页面只展示数据集身份、日期、价格口径和内容哈希，不提供下载、构建或发布按钮。

当前训练和普通／脚本回测默认读取本地 Parquet 快照，不调用封存的 `production.v2` 服务。数据根由 `QUANT_DATA_ROOT` 或既有路径 helper 选择；根目录保留的 `ACTIVE_RELEASE_SET.json` 不改变这项选择，也不需要删除。旧数据页的服务状态展示是独立兼容入口，不能用它判断本地回测是否可用。

回测按 ETF／A 股快照读取 qfq 行情，结果保存实际快照 ID、清单摘要、内容摘要和文件索引摘要；同次比较基准沿用已经选定的快照，不能在中途切换到新的 `LATEST`。只校验本次实际读取的行情文件，不把独立体检报告变成全库门禁。行情文件被替换时，已有计算缓存失效并重新核对文件摘要。起始日也是本次可用行情的起点，需要均线等预热数据的策略会在积累足够样本后才产生信号；比较旧 Run 时，须使用相同预热区间、执行规则和费用口径。

Qlib 继续使用已有 Direct Provider：冻结本地 A 股价格、历史成员区间、指数及适用清洗规则的身份，不复制整套行情为 bin，也不从供应商补历史。训练 Run 保留 descriptor SHA、股票与指数快照及配置／预测 artifact 身份。历史成员与指数的覆盖日期独立于股票行情；小规模跑通不代表最新日期的全部依赖已补齐。验证训练使用隔离的 reports／mlruns 和 `--no-publish-latest`，不会替换现有正式模型。

## 本地安全边界

本地研究默认只绑定 `127.0.0.1`。只要 `DSA_BIND_HOST`/`UVICORN_HOST` 指向非 localhost，或 `DSA_ENV=production`，后端就要求 `ADMIN_AUTH_ENABLED=true`；否则启动会直接拒绝。设置页也不能在这些暴露运行时关闭认证。需要局域网或公网访问时，先启用管理员认证，再配置明确的来源列表或反向代理。若绕过项目入口、手写 `uvicorn.Config(host=...)` 启动服务，必须先设置 `DSA_BIND_HOST` 或调用 `src.web_bind.publish_bind_host()`，否则应用工厂无法可靠感知真实绑定地址。

`/api/v1/quant/*` 的任务创建/取消/SSE、`/api/v1/backtest/*` 的回测写请求、`/api/v1/research/experiments/*` 的实验记录变更和 `/experiments/*` 的 MLflow 写请求，都要求请求带有本机 `Origin` 或本机 `Referer`。`/api/v1/quant/*` 的任务列表、日志、产物和状态读取、`/api/v1/research/status` 以及 `/experiments/*` 读取也会拒绝跨站浏览器访问；无 `Origin` 的本机 CLI 健康检查仍可读取。浏览器里的 DSA 页面会自动满足这个条件；裸 `curl`/脚本调试 mutation API 时，应显式带上本机 `Referer: http://127.0.0.1:8010/workflow` 或 `Origin: http://127.0.0.1:8010`。`X-DSA-Local-Client` 不作为来源保护旁路；一旦绑定到 `0.0.0.0` 或生产环境，应启用管理员认证后按受控客户端方式接入。

量化任务接口在本地模式下可以不启用登录认证，但安全边界是“只绑定 localhost + 来源校验 + 白名单脚本”。如果要给局域网或公网使用，必须启用 `ADMIN_AUTH_ENABLED=true`，并额外评估访问控制、反向代理和提交频率限制。

量化任务中心启动脚本时只向子进程传递白名单环境变量：`CHILD_ENV_EXACT_NAMES` 和 `CHILD_ENV_PREFIXES` 默认只包含路径、运行时和非敏感数据配置；`JQ_USERNAME`、`JQ_PASSWORD`、`TUSHARE_TOKEN`、`AKSHARE_TOKEN` 这类数据源凭据不会进入所有任务。只有任务 spec 显式声明 `env_names` / `env_prefixes` 时，对应 job 才能拿到必要凭据，例如 `download_jqdata` 只允许 JQData 登录变量。LLM/API provider key、`MLFLOW_URL`、`MLFLOW_*`、未列入白名单的 `QUANT_API_KEY` / `QUANT_SECRET_TOKEN` 和其他系统环境不会原样继承给 durable runner。新增任务脚本时也要遵守同一边界：不要把 token 放进命令参数，不要打印 `os.environ`，不要把密码写入日志或报告。

JQData 凭据默认从 `JQDATA_ENV_FILE` 指向的文件读取；未设置时依次尝试 `~/.agents/private/quant-workbench/jqdata.env` 和历史兼容位置 `WORKING_PROJECTS_ROOT/qlib/.env`。新安装和新 agent 维护时，应使用私有目录，不要把 `JQ_USERNAME` / `JQ_PASSWORD` 写回项目目录。

`MLFLOW_URL` 只用于网页链接和本地 health check。health check 只探测 `127.0.0.1`、`localhost` 或 `::1`，并拒绝带用户名密码或路径穿越段的 URL；API 返回给前端的展示 URL 会移除 username/password、query 和 fragment。非本机 MLflow 地址只展示为外部链接，不主动探测。`MAX_QUANT_JOB_STREAMS` 可调节 `/workflow` SSE 连接上限，默认 8，最小值按 1 处理。

本机 `MLFLOW_URL=http://127.0.0.1:6006/` 时，前端链接统一改写为 `/experiments/`。DSA 只缓冲 MLflow 的 HTML 首页以插入双语运行时；JS、CSS、API 和 artifact 按流转发。`/experiments/` 保留 MLflow 3.14.0 官方首页、实验、训练 Run、对比、参数、指标、artifact、Prompts、AI Gateway、Assistant 和 Settings 等完整导航与路由，不再隐藏或重定向非实验页面。翻译目录由官方 MLflow 3.14.0 `zh-CN.json` 和本机已安装的 3.14.0 前端 bundle 生成，共 1235 条静态映射；实验名、Run ID、参数名、指标名和文件名保持原文。部分页面仍可能残留少量英文，这属于已知显示差异。目录元数据中的 `upstreamCommit` 固定为 `86cd7f5a1dc25a1387ad07c87811bbc30f62951c`，生成器会拒绝版本不一致的 MLflow bundle。

当 `6006` 未启动时，`/experiments/` 离线页可以启动固定的本机 MLflow 命令。该入口不接收命令、路径或端口参数，只使用当前 DSA Python、仓内 `quant_workflow/mlruns`（或显式 `QUANT_WORKFLOW_MLRUNS_DIR`）、`127.0.0.1:6006`，日志写入 `logs/mlflow_6006.log`，启动记录写入 `.local/services/mlflow-6006.json`。子进程环境采用最小白名单，不继承 LLM API Key、JQData 密码等业务凭据。

任务状态 API 不返回真实 `cwd`、`job.log` 路径或 artifact 绝对路径；前端只拿 artifact 展示名和索引，下载/预览时仍由后端按 job id 与 artifact index 做路径边界检查。研究状态里的数据治理路径也只返回项目内标签或文件名，`.maintenance.lock` 只暴露锁定状态和更新时间，不暴露文件内容。

工作流页面的实时任务列表使用 SSE 长连接。直连本机通常会在浏览器关闭页面时释放连接；如果未来通过 Nginx、内网穿透或其他反向代理暴露页面，应配置合理的读超时和空闲连接回收，避免异常断线长期占用 SSE 连接槽。多 worker 部署下每个 worker 都有自己的 SSE 连接计数，因此有效上限约为 `worker 数量 × MAX_QUANT_JOB_STREAMS`；当前工作台按单机本地服务设计。

历史实验记录 API 的归档、永久删除和训练脚本追加记录共用 `quant_workflow/reports/.experiment.lock`。该 API 仅保留兼容，不再作为一级页面；正式用户入口统一查看 `/experiments/` 的 MLflow 记录。

## 项目职责

| 项目 | 当前角色 | 应该放什么 | 不应该做什么 |
| --- | --- | --- | --- |
| `daily_stock_analysis` | 主网页、API、用户操作台 | 页面、后端 API、受控任务调用、回测和结果展示 | 不保存真实密码/token，不下载或发布正式数据，不直接接券商交易 |
| `quant_data` | 受治理的数据仓库与本机数据服务 | 下载、清洗、质量验收、不可变快照发布和数据服务 | 不承担 DSA 页面、模型训练、策略回测或下单 |
| `quant_workflow` | DSA 仓内研究脚本和产物层 | 消费 `quant_data` 已发布快照，执行训练、回测和报告 | 不修补或重新发布 `quant_data`，不提交运行产物 |
| `Qlib` | 机器学习量化引擎 | 通过统一 Qlib Provider 读取 `quant_data`，执行选股模型训练和交易策略回测 | 业务层不按物理路径或存储格式决定数据是否可用 |
| `china-astock-quant` | 参考库和复用来源 | 由共享 DSA Python 环境调用其指标、样例策略、事件回测和工程经验 | 不接入交易模块，不改造成主网页 |
| `ashare-live-research` | 本机只读实时观察 skill/MCP | 同花顺、手机镜像、行情异动、辅助交易观察层的后续研究 | 不并入 DSA 主仓，不读取密码，不自动下单 |
| `a-share-us-catalyst` / CatDesk（九点猫研） | 外部催化观察服务 | 美股催化 A 股题材的独立页面和状态入口；负责发现题材和候选股 | 不把其候选信号历史验证混写成 Qlib 策略回测；不直接当买入信号 |
| 同花顺桌面端 | 后续只读观察方向 | 界面参考、人工核对、未来只读截图/OCR 研究 | 不读取账号密码，不自动下单，不连接券商交易接口 |

## 网页自助流程

### 1. 检查数据

打开 `/data`。页面应显示数据服务状态、正式数据集数量以及每个数据集的 `dataset_id`、`snapshot_id`、`manifest_sha256`、`content_sha256`、最新交易日和价格口径。数据服务未就绪时，页面必须明确报错，不能静默回退旧文件。

### 2. 开展量化研究

打开 `/research`。默认进入“选股研究”，集中显示选股模型训练、候选比较、实验任务和 **CSI800 选股信号稳定性检验**；切换到“交易策略研究”后，集中使用脚本策略、条件策略、选股模型与交易策略组合及 ETF 回测。所有任务只能调用后端白名单脚本，网页不能提交任意 shell 命令。

### 3. CSI800 选股信号稳定性检验

旧的“一次完成模型、策略、成本和确认期”的鲁棒开发流程已经结案。已完成的 504 个固定参数任务和 20 组 CatBoost 调参结果继续保留，但因为没有模型通过全部稳定性门槛，只能作为第零代失败基线，不能称为合格策略。结案证据由 `report_robust_strategy_closure.py` 写入原 Run 目录，不删除或改写原模型、预测和 MLflow 记录。

“CSI800 选股信号稳定性检验”先回答一个更小但更关键的问题：价量数据里是否存在跨季度、跨 504/756/1008 日训练窗口都稳定的选股信号。正式配置是 `quant_workflow/configs/robust_signal_research.yaml`，入口是 `launch_robust_signal_research.py`，报告独立写入 `quant_workflow/reports/robust_signal/`，MLflow 实验为 `robust_csi800_signal_research`。2018-2022 只用于预注册的八组信号筛选，2023-2025 用于 12 个季度的滚动样本外验收；任何预测产物只要包含 2026 日期就立即拒绝。主验收口径固定为 Top50，不能事后换 Top20、Top30 或 Top100 冒充通过。

`/research` 的“选股研究”中，“CSI800 选股信号稳定性检验”卡片可启动正式研究、请求“完成当前完整任务后暂停”，并从同一 Run 恢复；旧 `/workflow` 地址保留兼容。暂停不是强制杀进程：当前季度、窗口和模型必须先原子写完预测、模型、诊断和进度，然后 runner 才退出。恢复时会核对配置、Git commit、数据清单和 provider 上下文；任何一项变化都拒绝续跑。每个任务的因子诊断同时保存原始 Alpha158 数据和处理后的模型输入；后者包含板块标识与基于 CSI800 历史收盘价的四类市场状态 one-hot 特征。市场状态统一从配置冻结的 `2014-01-02` 起点连续计算，每个任务只读到自己的测试期末，并检查训练期之前至少有 60 个交易日用于预热；不能随 504/756/1008 日训练窗口改变计算起点。交易策略研究只展示已经落盘的因子质量、标签方案、TopK 敏感性、窗口稳定性和四种市场状态归因；状态归因先在全体日期上确定统一的标签/调仓取样日，再按状态拆分，不把 smoke 结果当正式研究证据。

该检验当前先执行价量信号阶段。若价量筛选或严格验证失败，状态为 `signal_diagnosis_complete`，下一门槛明确写成 `build_historical_fundamentals`；这表示“价量阶段完成、基本面阶段尚未执行”，不等于 `signal_not_found`。只有带真实公告日期、至少延迟一个交易日生效且覆盖率达到 98% 的基本面扩展也失败后，才能输出 `signal_not_found`。LightGBM、XGBoost、CatBoost 和 MPS DNN 只允许在 OLS/Ridge 已证明确定性信号存在后作为挑战者运行，不是首轮信号存在性的前置条件。

另设公开策略研究候选清单 `quant_workflow/configs/public_strategy_candidates.yaml`。用户提供的“两只 ETF 20 日动量轮动”已登记为 `etf_20d_two_asset_momentum_rotation`：它只比较 159915 和 513100，收盘计算信号、下一交易日开盘执行，不能与 CSI800 个股模型或 Qlib 选股结果混称。回测使用独立的 ETF 数据目录和报告目录；公开文章的收益数字只保留为未核验声明。该候选在基准成本口径下允许最大回撤 35%，这一放宽只适用于该 ETF 研究候选，不改变 CSI800 鲁棒策略的 20% 开发/确认门槛和 12% 纸面跟踪门槛。

三 ETF 防守轮动是独立版本：固定使用 159915、513100、512890，先筛选站上自身 20 日均线的 ETF，再选择 20 日涨幅第一，T 日收盘产生信号、T+1 共同开盘执行；没有参数搜索。离线研究写入 `quant_workflow/reports/etf_defensive_rotation/`，网页通过 `/backtest/etf-momentum?variant=three_etf` 查看。当前只消费 `quant_data` 发布但尚未正式验收的 51 标的测试快照，所以页面和 JSON 必须保持 `research_preview`、`usable_for_trading=false`，不能输出正式候选评级。ETF 下载、清洗、修补、快照发布和质量验收全部由 `quant_data` 负责；DSA 只严格核对清单与哈希后消费，不得自行补数据。打板轨道同样遵守这条边界，目前因缺少已验收历史分钟数据而暂停，只保留只读数据契约和离线研究骨架，不启动下载、训练或实时页面。

`/workflow` 的“鲁棒策略生命周期”是一条独立研究链路，不覆盖日常 `candidate_pipeline_latest` 或 `latest_training_result`。它以历史时点 CSI800 为股票池，要求按每个历史日期使用当时的 CSI300 与 CSI500 成分股，不能拿今天的成分股回看过去。历史成员默认先查询 JQData；如果历史权限不可用，才整批切换到 Baostock 的历史成分接口，实际来源、回退原因、指数代码和日期范围写入 `reference/pit_csi800/MEMBERSHIP_SOURCE.json`，同一份成员库不得静默混用两个来源。行情、成员区间、行业、ST、停牌、上市日期和板块元数据先通过固定门槛；成员区间覆盖率低于 99%、有效行情覆盖率低于 98%，或存在重复日期、未来数据、基准缺口时，状态必须是 `blocked_data`，训练不会启动。

网页按以下顺序使用：

1. “运行鲁棒策略开发流程”：执行数据预检、不可变 provider 准备、五类模型的多窗口/多随机种子滚动样本外训练，以及五套交易规则的低成本、基准成本、40bp 压力成本和延迟执行测试。`weekly_top50`、`low_turnover`、`regime_scaled` 和 `sector_constrained` 是用于衡量各模块作用的消融对照，只有同时包含低换手、市场状态减仓、行业与板块分散的 `full_risk_control` 可以成为冻结候选。五套规则都执行单股 3%、单一行业 25%、创业板与科创板合计 30% 的硬上限；卖出因停牌或涨跌停失败时，买入会按实际残留仓位重新计算容量和市场状态总仓位。市场状态归因使用前一交易日收盘后已知的状态，板块归因同时给出主板、创业板、科创板的平均/峰值仓位、组合收益贡献和相对 CSI800 基准的超额贡献；现金相对基准的贡献单独记录，板块与现金贡献按复利财富变化口径与组合总收益和总超额严格对账。开发期只允许输出 `development_failed` 或 `research_candidate_ready`。
2. “运行冻结策略 2026 确认期”：只对已经冻结的配置、Git commit、数据清单哈希、模型文件哈希和预测文件哈希运行一次确认。2026 结果不能用于回头换冠军；未通过时输出 `development_failed`。
3. “运行鲁棒策略当日纸面跟踪”：只加载冻结模型做当日推理，不重新训练，按交易日逐日记录任务成功、组合收益、基准收益、净超额和最大回撤。状态 `shadow_observation` 表示仍在观察，不等于合格；完成配置中冻结的 60 个交易日且全部门槛通过后，才允许出现 `robust_strategy_qualified`。开发期、2026 确认期和纸面跟踪的所有数值门槛均读取同一份 `robust_strategy.yaml`，判定结果会保存实际使用的门槛，不使用脚本内另一套隐藏数字。

当前人工控制边界：旧的完整鲁棒策略 runner 仍只有命令行 `--resume`，没有网页安全暂停；CSI800 选股信号稳定性检验 runner 已支持网页协作式暂停和同一 Run 恢复。两条流程不能混称为都已支持。该检验的按钮只管理 `robust_signal` Run，不会停止、恢复或覆盖旧 `robust_strategy` Run。

有界调参仍限制为每个失败模型最多 20 组、最多 3 个完整矩阵入围者。筛选阶段必须覆盖 504/756/1008 三个训练窗口，并在配置指定的 756 日参考窗口覆盖 42/2026/3407 三个随机种子；候选排序先看原有稳定性门槛，再看 RankIC。不能恢复成只跑 756 日、只按 RankIC 选前三名的旧逻辑，因为那会在评估稳定性之前淘汰可能更稳定的参数。

开发期一旦产生 `research_candidate_ready`，会冻结入选模型在全部 12 个季度、全部训练窗口和全部随机种子上的模型文件与预测文件。固定参数入选模型只能绑定固定参数矩阵；经有界调参入选的模型只能绑定最佳调参配置重新训练得到的完整矩阵，不能把调参参数和固定参数模型混在同一冻结记录中。冻结记录保存本次开发 run 目录、每个任务的时间切分、上下文哈希、绝对路径和 SHA256，并对完整产物矩阵再计算总哈希；2026 确认期会在开始前和发布结果前核对这套冻结上下文，不能从其他 run、共享 `latest` 文件或后来改写的模型文件中替换证据。三个指数基准必须覆盖 2014 年以来每个必需交易日。历史指数在成分退出和替补生效之间可能短暂少于名义数量；构建器只接受单个指数和 CSI800 并集均不低于名义数量 99% 的原始记录，保留短缺异常，不用前向或后向填充伪造成分，低于 99% 时立即阻断。最终预检仍要求至少 99% 的交易日达到完整 800 只。

固定状态只有 `blocked_data`、`development_failed`、`research_candidate_ready`、`shadow_observation`、`robust_strategy_qualified`。网页会把策略生命周期和最近一次纸面任务分开显示：某天推理失败会显示日期和原因，但不会把一次运行故障伪装成 60 日最终结论；同一失败日期也不能重跑成成功。`/backtest` 展示模型稳定性、策略成本/延迟压力、季度样本外结果、市场状态归因、确认期和纸面跟踪；`/experiments/` 查看 `robust_csi800_development` 与 `robust_csi800_confirmation` 的 MLflow 原始 Runs。

回测市场面板从证券主表读取 `list_date`，并从每只股票的完整日线按真实有成交记录计算 `listing_trading_day`。科创板、注册制创业板和注册制主板上市前五个交易日，以及北交所首个交易日，按当时无涨跌幅限制处理；注册制前主板上市首日保留上涨 44%/下跌 36% 的方向差异，成熟科创板和注册制创业板风险警示股票仍按 20% 处理。价格文件若在上市后 30 天内才开始且缺失上市首日，市场面板直接拒绝；对于远晚于上市日才开始的旧历史文件，只标记为成熟股票，不得误判成上市前五日。缺少上市日期或可交易日年龄时回测拒绝继续，不用自然日或默认 10% 猜测。

正式报告写入 `quant_workflow/reports/robust_strategy/`。核心入口包括 `build_pit_csi800.py`、`download_robust_reference.py`、`validate_robust_price_bundle.py`、`import_robust_price_bundle.py`、`build_robust_price_provider.py`、`preflight_robust_data.py`、`prepare_robust_provider.py`、`run_robust_development.py`、`run_robust_strategy.py`、`run_robust_confirmation.py`、`run_robust_shadow_day.py` 和 `robust_shadow_tracking.py`。远端或离线生成的全历史 K 线必须先由 `validate_robust_price_bundle.py` 逐文件检查股票集合、PIT 成员快照哈希、日期、OHLCV 和 SHA-256；传回本机后由 `import_robust_price_bundle.py` 复验并发布到哈希隔离的只读运行目录，不直接覆盖正式 `quant_data/parquet`。`build_robust_price_provider.py` 再对只读价格包执行异常扫描和清洗，按已验证的完整基准日历生成 Qlib 特征，并事务导入 CSI300、CSI500、CSI800 三个基准目录；输出仍按内容哈希隔离，不覆盖日常 cleaned provider。`sync_data_manifest.py` 只在 `latest_product.json`、`BUILD.json`、`CONTENT.json` 和实际文件集合相互一致时登记 `robust_price_product`，否则固定记入 `pending_entries`，避免训练配置指向未完成或路径漂移的产品。网页和命令行的长开发流程统一通过 `launch_robust_strategy.py` 注册为当前 macOS 用户的 `launchd` 一次性任务；监督进程先用 `/usr/bin/env -i` 清空图形会话继承环境，再只传递非敏感运行白名单，主流程不会继承 GitHub/LLM/API token。启动命令必须显式固定 `--mlruns-dir` 到 DSA 正式 `quant_workflow/mlruns/`，不能写入隔离 worktree 的空目录。监督进程记录主流程 PID、标准输出、错误日志、launch record、退出码和最终状态，并在启动后复核进程与 CPU 占用。历史股票池每完成一个完整年度就事务提交该年度分区和进度；总成员区间只在全部年度完成后发布。`.build_pit_csi800.in-progress.json` 存在期间，读取方固定拒绝半成品。`/workflow` 从当前 run 的专属报告显示 `completed_date_count / expected_date_count`，不读取其他 run 的共享旧进度。不能把正式训练作为 Codex 前台子进程裸跑。模型、行情、报告、MLflow 数据和运行日志不进入 Git。

在真实数据预检通过之前，只能说“实现已就绪”或“数据阻断”，不能说训练完成；开发期与确认期通过后也只能叫“研究候选”，不能说“鲁棒策略合格”。整个流程不连接券商、不读取账号密码、不自动下单。

### 2.2 quant_workflow 脚本覆盖盘点

`quant_workflow/scripts` 里的稳定脚本不等于全部都应该直接放到正式用户页面。一级入口只在 `/research` 展示研究与回测任务；旧 `/workflow` 继续承载历史兼容和内部运行管理，但不进入一级导航。数据下载、清洗、质量验收、Provider 构建和快照发布归 `quant_data` 所有，不能重新放进 `/data` 或其他正式 DSA 入口。

### 2.3 每日数据更新与原生 V2 校验

`run_daily_update.py` 在 `/workflow` 中显示为“运行每日数据更新与 V2 校验”。它不再生成旧 merged/cleaned 候选，也不再调用 `publish_daily_candidate.py`：

1. 可选 A 股来源刷新默认关闭。显式选择 Baostock、东方财富或 AkShare 时，一次只运行一个来源，并沿用至少 2 秒间隔、1 次尝试、连续失败最多 3 次和最多 7 天重叠窗口。这些结果只进入审计缓存，不能发布正式 V2；明确选择的来源失败时本轮返回非零。
2. 全球日线、限定范围的 A 股新闻和 ETF 分别更新。ETF 通过 `quant_data_cli.py update cn_etf_daily` 整包原子发布；raw、qfq和因子位于同一个快照，失败不会切换正式指针。更新后自动运行 `quant_data_cli.py validate cn_etf_daily --release`，只用一条离线命令验收当前快照、唯一备份、根清单和策略标的，不扫描无关A股数据；只有更新和验收都成功才生成新ETF报告。任一步失败时保留上一份报告并继续A股主流程。
3. 正式 A 股更新只能由已核验的 raw/factor/qfq bundle 触发。命令行显式传 `--ashare-v2-bundle-dir <bundle>` 后，编排器委托 `quant_data_cli.py update ashare_daily_qfq --bundle-dir <bundle>`；该事务构建新的 `ashare_daily_raw`、`ashare_daily_qfq` 和 `qlib_ashare_v2` 不可变快照，并在失败时恢复原指针。网页不接受任意本机 bundle 路径。
4. 正式数据更新、质量检查和原子发布全部由 `quant_data` 完成。DSA 不再通过 `report_data_quality.py` 重复验收或授权通用训练；只实时读取 `quant_data /ready`、检查任务所需正式数据集，并确认当前 Qlib Provider 可读且血缘完整。旧编排器和历史报告仅作兼容与既有 Run 复现，不作为新的用户操作入口。
5. `official_published=true` 只表示本轮 V2 bundle 事务真实成功；普通审计缓存下载成功不会设置该标志。运行摘要同时记录 qfq、Qlib V2 和 ETF 更新前后的日期以及 `legacy_source_role=audit_cache_only`。

`--skip-download` 表示跳过独立下载、只校验现有 V2；`--dry-run` 只列出计划且不启动子进程。固定路径兼容输出只能由 `publish_v2_compatibility.py` 维护；mutable 修复工具不得写入原生快照。

JQData 不纳入每日自动下载：其账号数据权限窗口结束于 Baostock 长历史序列之前，只在合并和跨源验证中复用既有 JQData parquet。东方财富和 AkShare 是免费冗余源，首次全市场下载会额外占用约 1GB 量级磁盘；日常增量只补最近窗口。

东方财富、AkShare 和新浪快照任务默认绕过 macOS/系统代理，避免本机代理端口残留时把公开行情下载误判为源站失败；确实需要代理访问时，在网页任务里勾选“使用系统代理”，或在脚本端传 `--use-system-proxy`。Baostock 阶段使用其 SDK 的匿名连接路径，不走这组 requests 代理开关。

外部端口被封、上游数据源拒绝连接或本机网络暂时不可达属于运行环境状态，本身不作为 DSA 软件缺陷或代码审查阻塞项。DSA 必须把这类失败明确显示为外部连接失败，并允许重试或切换已批准来源；误报成功、写入半套数据、破坏现有快照、无限等待或无法恢复才属于产品缺陷。测试和审查不得为了让外部连接变绿而降低数据验收门槛。

### 数据验收与训练消费边界

`quant_data` 负责下载、清洗、完整质量验收，以及发布带内容身份的不可变数据集和 Qlib
Provider。DSA 的通用训练入口不再重复运行或读取完整质量报告来决定训练权限；提交前只做
轻量预检：`quant_data /ready` 可用、任务所需正式数据集存在、所选 Qlib Provider 可读，
并且 Provider 的不可变血缘完整。历史 `data_quality_report_*` 继续用于审计，但不能授权或
阻止通用训练。训练模块不下载数据、不清洗数据，也不重新发布数据。

DSA 的研究页、数据页和普通状态 API 不展示历史质量报告、上一验收快照或重新验收入口。
旧质量字段暂时保留为空值或 `deprecated`，磁盘上的存量报告暂不删除，以免破坏既有 Run
的冻结复现证据。数据质量检查只属于 `quant_data` 的发布过程；CSI800 已启动 Run 继续使用
自己的冻结证据和 `robust_signal_data_v3` 专用准备契约。

CSI800 选股信号稳定性检验另有专用冻结输入契约。数据模块把每次验收结果写为
`readiness_artifacts/robust_signal_data_readiness_<sha256>.json`，再原子更新
`robust_signal_data_readiness_latest.json` 指针。网页提交新 Run 时只解析一次 latest，
把不可变文件路径和 SHA-256 写进持久化任务命令；训练运行器只消费这份冻结文件，并验证
readiness 必须位于受控的
`quant_workflow/reports/robust_signal/readiness_artifacts/`，不能用其他位置的同名目录
冒充。readiness 同时记录数据模块实际发布的质量证据文件名和 SHA-256；训练运行器只在
受控 `quant_workflow/reports/` 根下读取该冻结证据，核对证据契约、文件哈希、cleaned
provider 身份、provider 元数据、provider 内容身份和历史 CSI800 成员区间身份。它不
读取当前 `data_quality_report_latest.json`，也不重新计算覆盖率、质量阈值或训练资格。

正式 CSI800 选股信号稳定性检验只接受 `robust_signal_data_v3`、
`acceptance_owner=data_module` 和
`acceptance_contract=data_module_accepted_content_addressed_snapshot_v2`。旧版 v1/v2
正文即使手工改字段，也不能冒充 latest 指针。数据更新后由数据模块发布新验收文件；
已经开始的 Run 继续使用自己的旧哈希，恢复任务也从原 `context.json` 取回该哈希。
需要使用新数据时，应创建新 Run。外部网络失败本身只是环境事件；
只有 DSA 误报成功、发布损坏或不完整数据、无限挂起、恢复失败或绕过该专用冻结数据契约，才算软件缺陷。

### 本轮已修复的四个非阻塞 P3

1. 未加载 Run 时，回测日期、TopK 和成本输入框与执行按钮同时禁用。
2. 训练结果和回测详情中的 MLflow Run ID 直接链接到对应实验和 Run。
3. SSE 事件 JSON 损坏或缺少 `jobs` 数组时记录诊断警告，不再静默吞掉事件。
4. 研究状态刷新失败时清除旧状态，避免页面继续展示过期训练信息。

独立复审进一步发现并已纳入本轮修复：候选模型目录使用模型级 MLflow
`experiment_id` 生成 Run 深链；CSI800 专用 readiness 类型保留验收所有者、验收契约、
冻结证据、readiness 文件名/SHA-256 和 cleaned provider 身份字段；训练任务提交和直接
CLI 运行都拒绝受控报告根之外的 readiness 文件。这些专用字段不回流为通用训练门禁。

以下两项仍是明确记录的后续维护项，不阻碍当前单机训练：任务列表逐行执行
`ps`/glob/stat 的性能优化，以及未来多 worker 部署时为通用任务更新增加 SQL 状态守卫。

自动调度用本机 launchd（不写入 Git，属机器本地配置）。安装脚本从仓内模板生成含绝对路径的 plist：

```bash
scripts/launchd/install-quant-daily-update.sh            # 生成 plist 并打印加载命令
scripts/launchd/install-quant-daily-update.sh --load     # 生成并立即 bootstrap
scripts/launchd/install-quant-daily-update.sh --dry-run  # 只打印解析后的 plist，不写入
scripts/launchd/install-quant-daily-update.sh --uninstall # bootout 并删除 plist
```

任务 `com.dsa.quant-daily-update` 的配置仍保留工作日 18:40（Asia/Shanghai），但当前本机因 Baostock `10001011` 和东方财富断开问题已执行 `launchctl disable` 并卸载，不会自动触发。恢复时应保持 `--ashare-source none`，只运行独立数据更新和原生 V2 校验；不得因为换了编排器就重新开启被暂停的 Baostock/东方财富自动请求。任务本身带单实例锁；DSA 网页还会阻止不同网络下载按钮并发运行。

四个直接脚本 `download_baostock.py`、`download_eastmoney.py`、`download_akshare.py` 和 `download_ashare_news.py` 共用 `source_download.lock.json`；即使绕过网页从两个终端同时启动，第二个进程也会在发送网络请求前退出。
模板默认让 `com.dsa.quant-daily-update` 在工作日 18:40 运行，用 `daily_stock_analysis/.venv` 执行仓内 `run_daily_update.py`，日志写入 `quant_workflow/logs/`。这个任务只更新数据，不运行 60 日纸面跟踪。

2026 确认期通过后，另行安装纸面跟踪任务：

```bash
scripts/launchd/install-robust-shadow.sh --dry-run  # 核对 Python、配置和 19:10 调度
scripts/launchd/install-robust-shadow.sh --load     # 确认期通过后才安装启用
scripts/launchd/install-robust-shadow.sh --uninstall
```

`com.dsa.robust-shadow` 在工作日 19:10 启动，并最多等待 4 小时让 18:40 数据任务结束。当天数据更新失败、只是 dry-run、等待超时或推理失败时，该任务返回非零并写 `robust_strategy/shadow_runner/schedule_latest.json`，不能降级成日更新 warning。冻结组合含 `mps_dnn` 时，安装脚本默认使用 `.venv-mps-dnn`，每次推理前保存真实 MPS 探针；不允许自动回退 CPU。尚未通过确认期时任务只返回 `inactive/skipped`，不会伪造纸面观察。正式启用前还必须确认 18:40 数据任务已经恢复成功；旧流程停在 `apply_data_cleaning` 的记录不能据此开始 60 日计数，新流程必须以原生 V2 校验阶段全部通过为准。

### 2.3 只读盘点与报告任务

`report_data_hygiene.py` 在 `/workflow` 中作为“生成数据治理盘点和清理预案”任务出现。它只读检查 `quant_data`、`quant_workflow`、DSA、Qlib、china-astock-quant 的当前体量、任务库状态、业务进程、官方 parquet、官方 Qlib provider、根目录日志/临时脚本和历史报告，输出 `data_hygiene_inventory_latest.{md,json}` 与 `data_hygiene_cleanup_plan_latest.{md,json}`。清理计划只包含 `keep`、`document`、`move`、`archive` 等 dry-run 建议，不包含真实删除动作；所有会改变路径的建议都标记为需要用户确认。

`report_workbench_capabilities.py` 在 `/workflow` 中作为“生成工作台能力盘点”任务出现。它只读扫描 DSA 前端页面、DSA API router、后端量化任务白名单、网页任务卡、`quant_workflow/scripts` 稳定脚本、测试覆盖和 `china-astock-quant` 可复用边界，输出 `workbench_capability_inventory_latest.{json,md}`。这个任务用于回答“哪些能力已经有网页入口，哪些还缺测试、缺前端入口或只保留在脚本层”，不启动下载、转换、清洗、训练任务，也不接券商、不自动下单。

### 2.4 鲁棒选股信号长训练控制

`/research` 的“选股研究”已经为 CSI800 选股信号稳定性检验提供受控网页入口：预览固定研究矩阵、运行隔离的单季度测试、正式启动、完成当前原子任务后暂停、按同一 Run 恢复和立即取消。网页显示 Run ID、阶段、完成数、总数、百分比、暂停状态、检查点以及配置、代码和数据身份；浏览器不能提交任意命令、输出路径或 launchd label。

- **协作暂停**只写入暂停请求，runner 完成当前原子任务并保存检查点后进入 `paused`。这和立即终止进程不同。
- **网页恢复**只接受原 Run ID；恢复前重新校验配置哈希、Git 提交、数据清单、provider 树和已有产物。身份变化时失败关闭，不会悄悄新建 Run 或从头训练。
- **检查点完整性**不再以“文件存在”作为完成依据。每个已完成子任务保存固定的 Run 内相对路径、SHA-256、字节数、任务身份、阶段、信号、选股模型、季度、训练窗口、时间切分、行数和 artifact schema；恢复时重新读取同一份预测 Parquet 字节校验字段与锁定日期范围，并校验 diagnostics JSON。模型 pickle 只核对受控路径、哈希和模型契约，不为检查点验证而反序列化。
- **立即取消**调用服务端白名单中的专用取消命令，并核对 launch token、PID、进程启动身份和命令哈希。只有后台 runner 确认停止后才标记 `cancelled`；取消失败会保留可见错误，不能伪装成已停止。
- **启动授权**采用两阶段协议。任务必须先在共享 SQLite 中原子完成 `pending -> running` 认领；耐久 wrapper 只登记签名身份并等待一次性许可，服务端再次核对任务状态、launch token 和 PID 后才允许执行真实命令。另一个 DSA 进程先取消时，任务不会被复活，真实训练命令也不会启动；授权先完成时，已登记的进程组可被准确终止。
- **启动器与 runner 分离**：launchd 接管训练后，网页启动任务本身可以先显示 `completed`。只要同一个 Run 的持久化状态仍是 `starting`、`running` 或 `pause_requested`，API 仍返回 `can_cancel=true`，页面继续提供“立即取消”；自然完成或重复取消不会再次执行取消命令。
- **重启恢复**：API 重启时会等待严格的 runner identity 握手，并验证 token、PID、命令哈希、进程启动时间和实际进程命令。身份文件仍在生成时保留输出冲突锁；握手超时只终止本次新建且身份精确匹配的进程，不会误杀其他训练。
- **恢复监控超时**：已验证仍存活的 runner 不会仅因监控窗口到期被标记为失败。页面显示告警，任务继续占用冲突产物并重新挂接监控，直到取得真实退出状态或完成经过身份核对的取消。

以上能力只覆盖 `run_robust_signal_research` 和 `resume_robust_signal_research`。旧鲁棒策略开发 runner、确认期、60 日纸面跟踪和 MPS 专用训练没有因此自动获得同一套网页暂停/恢复功能；这些流程仍按各自持久化 runner 和维护文档执行。立即取消也不是可恢复暂停，页面会二次确认并明确提示这个区别。

### 2.5 网页选股模型训练闭环

`/research` 的“选股研究”主体是选股模型训练，不把选股模型和交易策略混成一个概念。用户可在 `/data` 查看 `quant_data` 已发布快照，再返回选股研究；页面显示当前 Qlib Provider 身份和轻量训练预检。只有 `quant_data /ready` 不可用、必需正式数据集缺失、Provider 不可读或不可变血缘不完整时，通用正式训练才会阻止，按钮旁必须显示具体原因。页面不再要求用户重复运行“完整质量验收”。

CSI800 选股信号稳定性检验仍使用前述专用 `robust_signal_data_v3` 冻结 readiness。它与通用训练轻量预检不是同一契约；通用 Provider 预检通过，不代表 CSI800 专用 PIT 股票池和冻结输入已经就绪。

页面请求数据状态期间必须显示中性的“读取中”，不能在结果尚未返回时提前显示“阻塞”。当前 DSA Web 的正式设计与验收范围仅覆盖桌面端；现有移动端自适应行为可以保留，但不作为功能目标、兼容性承诺或 Agent C 复测条件。

测试式训练与正式训练使用不同的产物和实验边界：

- 候选流程 dry-run 只写 `candidate_pipeline_test_*` 和测试进度，页面统一标记“测试数据，不可用于研究或交易”。它不覆盖 `candidate_pipeline_latest`，不进入正式候选列表，也不创建正式 MLflow Run。
- 真实网页/脚本等价验收使用专用配置、报告目录和 MLflow 实验 `dsa_training_equivalence`，不替换日常 `latest_training_result`。验收必须固定同一配置、provider snapshot、随机条件和时间区间，并比较两个真实 Run 的预测哈希、关键指标、MLflow 归属及回测结果；dry-run 不能证明等价。
- 正式训练只能由网页提交到后端白名单任务和持久化 runner。网页不接受任意命令、脚本、输出路径、MLflow 实验名或 provider 绝对路径；终端只用于诊断、测试和只读核对。

训练结果是否能进入 `/backtest` 由回测契约决定，不能只看模型名称或是否有 Run ID。可回测结果必须同时具备真实 Run ID、可重放预测、完整不可变 provider 身份以及冻结的交易策略配置。训练结果按钮会把 Run ID、选股模型、交易策略、snapshot、日期、TopK 和成本写入回测 URL；回测页在同一个已挂载页面内切换这些查询参数时也会重新同步选择并清除旧结果。URL 没有 Run/模型身份时才允许默认选择第一项；URL 明确指定的 Run、模型或组合不存在、不匹配时，页面必须清空旧选择和旧结果并显示错误，不能静默改成另一条 Run。测试结果或缺少契约的历史结果只提供“查看测试结果”或缺失原因，不显示“运行回测”。回测目录 API 可以保留旧记录用于审计，但前端会把 `replay_capability=unavailable` 的 Run 排除在可操作下拉框之外，并汇总显示排除数量和原因；`custom_only` 且仍有完整数据身份和冻结策略的旧 Run 继续允许明确标记的自定义回测。

候选流程的子训练记录使用 `candidate_child` scope。baseline 目录跳过这些子记录，由候选报告按真实模型 ID 只生成一条目录项，避免同一 MLflow Run 以 `latest` 和候选模型两个身份重复出现。候选 DSA 交易策略依赖当前 `OptimizedStrategy` 引擎，因此统一标为“重建回测 / custom_only”，不再宣称“按原条件重放”。baseline 和 candidate 的所有可运行重放都必须使用选中模型条目中冻结的 `pred.pkl`、`label.pkl`（Qlib exact 另含 `task`）哈希；服务只读取 artifact 一次，比较同一份字节的 SHA-256 后再从该字节反序列化，不能先验证路径再重新打开文件。

任务状态同时区分短生命周期启动器和实际 durable runner。launchd 已接管但启动器任务先显示 `completed` 时，只要 `can_cancel=true`，页面仍将其计入运行中任务、阻止重复启动并显示“后台仍运行”。顶部计数、任务分组、日志、进度和启动按钮统一使用这一判断。页面展示前端构建、后端构建、API 契约、runner PID/存活状态、当前阶段、完成数、总数、百分比、已用时间、可计算时的预计剩余时间、当前产物和失败阶段；日志截断会明确标记。

市场状态与现金仓位报告只接收最近一次正式全量质量报告批准的 A 股 qfq 快照。质量报告已经完成内容哈希验证；网页提交任务时再比较该快照的文件系统状态指纹，防止验收后、进程启动前数据被替换，同时避免每次报告都重新读取整套行情文件。

当前已知边界必须如实保留：旧候选 Run 如果没有不可变数据身份或冻结策略，只能查看缺失原因；旧版但具备数据身份和策略的 Run 可能只支持自定义回测，不能称为原条件重放。CSI800 选股信号稳定性检验的预览、测试、正式启动、暂停和恢复按钮都必须等待并遵从后端任务目录；后端未加载、版本不兼容或未启用时显示具体不可用状态，不发送请求。通用 LightGBM 正式训练与严格鲁棒 CSI800 研究是两套不同的数据准备契约；通用 provider 可训练不代表鲁棒 PIT 数据已就绪。旧鲁棒策略 runner、确认期和纸面跟踪仍没有同等级的网页协作暂停/恢复。系统不连接券商、不自动下单。

首页提供 CatDesk 9（`a-share-us-catalyst` / 九点猫研）的本地服务入口。点击后由 DSA 先检查固定本机服务；未启动时只允许执行仓内约定的 CatDesk 启动程序，按钮显示启动状态，健康检查通过后再打开 `http://127.0.0.1:8765/`。该服务不共享 DSA/Qlib 数据，也不把 CatDesk 的“候选信号历史验证”归类为 Qlib 策略回测。首页的 MLflow 入口采用同样的固定服务检查与启动流程，健康后打开 DSA 的 `/experiments/` 代理页。

`report_catdesk_signal_validation.py` 是 CatDesk 候选池的研究验证脚本。单日模式未填写验证日时，脚本会从 cleaned Qlib provider 的 A 股交易日历自动选择信号日后的第一个交易日；显式填写的日期不是下一交易日时会直接拒绝，避免把多日收益冒充一日验证。日历缺失或日期超出本地日历范围时才退回“只跳周末”的规则，并在 JSON 和 Markdown 报告中标记 `weekday_only`。默认严格口径下，Qlib run 必须早于验证日 09:30，脚本才会把 Qlib 分数纳入排序；晚于这个时间的模型分数会标记为不可用，避免把事后生成的分数当成事前信号。若只是做事后研究对比，可以显式传 `--allow-backfilled-qlib`，报告会记录 `qlib_backfilled_allowed=true`，意思是“用了事后补跑的 Qlib 分数做研究比较”，不是证明当时盘前已经有这个信号。

历史验证还会拒绝信号日 09:30 之后生成的同日 market-regime 文件，并回退到最近一个更早的安全文件；严格 Qlib cutoff 非空但无法解析时直接标记 Qlib 不可用，Markdown 会明确说明 Qlib 未参与调整。验证日在线快照只有 `trade_time >= 15:00:00` 才作为最终收盘证据，盘中或时间不可解析的快照会记录 `validation_status=provisional_intraday`，其 `validation_return_pct` 为 `null`，不进入命中率、收益统计或实际赢家排行；实际赢家只接受有限数值收益。滚动模式按每组交易日自动推导验证日，因此不会读取或校验单日模式的 `--validation-date`。

`report_china_astock_indicators.py` 复用 `china-astock-quant/strategies/_indicators.py` 的 SMA、EMA、RSI、ATR 和 Bollinger 指标函数，读取本地 cleaned 行情生成 `china_astock_indicator_report_latest.{json,md}`。这是只读研究报告，不连接券商、不读取账号密码、不自动下单。网页端和脚本端省略 `limit` 时都默认只抽样 50 只股票，避免误把“安全报告”跑成全市场重任务；需要扩大范围时必须显式填写数量上限。

`report_qlib_research.py` 在 `/workflow` 中作为“Qlib 研究增强”的 6 个固定任务出现：选股模型效果深度解释、交易策略回测详细分解、多选股模型/多实验对比、因子挖掘辅助、滚动训练计划和组合风险诊断。每个网页按钮都固定传入对应 section，不能从网页提交任意 section 或 shell 命令。脚本读取本地 MLflow/Qlib 既有产物生成 `qlib_research_model_explainability_latest.{json,md}` 等报告；首版不重新训练、不计算无限制 SHAP、不接券商、不自动下单。

`run_candidate_pipeline.py` 是完整候选流程的固定编排入口，配置真源为 `quant_workflow/configs/candidate_pipeline.yaml`。五个选股模型使用同一个 Alpha158、CSI300、训练期、验证期、交易成本和基准；选股模型与交易策略组合的排名只读取 2024-2025，2026 指标只用于对已选组合做独立终审，不能根据 2026 成绩改选其他组合。每轮写入 `candidate_pipeline_latest.{json,md}`、进度 JSON 和同批次 MLflow runs；真实候选 Run 会保留在实验历史中以便追溯，但候选研究结束后会恢复运行前的 `latest_training_result.json`，不会自动替换日常选股模型、连接券商或下单。

研究状态 API 只会对配置的 `quant_data/qlib` provider 根目录内的 Qlib 路径补充日历、标的和体量信息。`latest_training_result.json` 如果指向根目录之外的路径，页面会保留警告，不递归扫描该目录。

`add_index_benchmark.py`、`generate_csi300.py`、`repair_qlib_provider.py`、`supervise_baostock.py` 等旧数据维护脚本只保留历史兼容，不进入正式用户页面。数据下载、清洗、Provider 构建和快照发布由 `quant_data` 项目负责；DSA 不通过其他入口重新暴露这些操作。

### 4. 查看回测和实验

在 `/research` 切换到“交易策略研究”。旧 `/backtest` 地址继续兼容。

- `脚本回测`：直接编写 DSA Python 策略。系统先做静态安全检查，再放入有超时和资源限制的子进程；脚本只生成目标仓位，实际成交、成本、现金、净值和回撤由公共引擎计算。
- `条件回测`：不用写代码，选择动量轮动与均线过滤、双均线择时或等权买入持有，并填写标的和参数。
- 两种模式都只读取 `quant_data` 中 `canonical + ready + qfq` 的日线快照；股票和 ETF 使用同一引擎。网页默认成本为 `0%`，只有用户输入的成本才参与计算。
- Qlib 的 Run、参数、指标和 artifact 改在 `/experiments/` 查看。旧 DSA 历史判断复盘和历史信号 API 继续兼容，但不再作为 `/backtest` 的子选项。
- `Qlib 预测摘要`：读取最新 MLflow `pred.pkl`，展示最新交易日的最高/最低模型评分，并写入 `qlib_predictions_latest.{json,md}`；它是只读排行，不是自动买卖信号。
- `Qlib 研究增强`：读取最新或指定 MLflow run 的 `pred.pkl`、`label.pkl`、`params.pkl` 和 `portfolio_analysis`，生成模型解释、回测明细、多实验对比、因子 IC、滚动训练计划和风险诊断报告。
- `实验复现与产物`：在 `/experiments/` 核对本次训练使用的数据版本、provider、source config、runtime config、MLflow run id、参数、指标和关键 artifact，例如 `pred.pkl`、`portfolio_analysis`、`task`。
- `交易策略优化预演`：在受控研究任务中用内置小样本 dry-run 验证低换手、市场择时和回撤保护实验链路；它只读生成报告，不训练选股模型、不连接券商、不自动下单。
- `ETF 固定规则报告`：`/backtest/etf-momentum` 可在旧两 ETF 报告和三 ETF 防守轮动研究预览之间切换。三 ETF 页面只读取已有报告，不从网页触发下载或重新计算；未验收快照上的最新信号始终标记为不可用于交易。
- “按原条件重放”会在本机重新运行冻结的组合回测，冷启动可能超过 30 秒。前端只对该本地计算使用 120 秒等待上限；若仍超时，页面会说明“本地回测等待超时”，不能把它误报成外部行情、代理或网络故障。
- `/workflow` 的任务轮询使用 `/api/v1/quant/status?include_research=false`，避免和独立研究状态接口重复组装大数据快照。需要完整状态时，后端先完成研究数据和本地服务探测，再读取可变的任务状态；因此任务在慢查询期间完成时，响应不会继续显示请求开始时的旧“运行中”状态。

从首页打开 `/experiments/` 查看 MLflow 原始实验记录。历史 DSA 实验记录管理 API 继续兼容旧调用方，但不再作为并列用户入口。

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
| 量化报告和实验记录 | 默认 `daily_stock_analysis/quant_workflow/reports/`；旧同级 `quant_workflow/reports/` 不再自动兜底；可用 `QUANT_WORKFLOW_REPORTS_ROOT` 显式固定位置，运行态产物不入库 |
| MLflow 原始记录 | 默认 `daily_stock_analysis/quant_workflow/mlruns/`；旧同级 `quant_workflow/mlruns/` 不再自动兜底；可用 `QUANT_WORKFLOW_MLRUNS_DIR` 显式固定位置，运行态产物不入库 |
| 临时实验草稿 | `daily_stock_analysis/quant_workflow/scratch/`，只保留 README |
| Qlib provider 和行情数据 | 默认优先 `daily_stock_analysis/quant_data/`，若仓内不存在则使用 DSA 同级 `quant_data/`；可用 `QUANT_DATA_ROOT` 显式固定位置。运行态数据不入库 |
| CodeGraph 索引 | 每个项目自己的 `.codegraph/` |
| Python 虚拟环境 | `daily_stock_analysis/.venv/` 是唯一 Python 运行时；`qlib/.venv` 和 `china-astock-quant/.venv` 不作为网页任务默认运行环境 |

可迁移路径统一通过环境变量覆盖：`QUANT_DATA_ROOT` 指向行情数据和 Qlib provider 根目录，`QUANT_WORKFLOW_ROOT` 指向量化脚本层，`QUANT_WORKFLOW_REPORTS_ROOT` 指向报告/实验摘要，`QUANT_WORKFLOW_MLRUNS_DIR` 指向 MLflow file store，`MLFLOW_URL` 控制网页里展示的 MLflow 链接，`MAX_QUANT_JOB_STREAMS` 控制工作流实时状态连接上限。默认代码真源和运行态正本都是仓内 `quant_workflow/`；旧同级 reports/mlruns 不再自动读取或写入，迁移旧历史应复制到仓内目录，临时回滚才显式设置环境变量；其余默认值分别是仓内或同级 `quant_data/`、`http://127.0.0.1:6006/` 和 `8`。

量化任务统一使用 `QUANT_WORKFLOW_PYTHON`，默认指向 `daily_stock_analysis/.venv/bin/python`。旧的 `QLIB_PYTHON` 不再作为网页任务中心的 Python 入口；如果其他脚本仍设置了 `QLIB_PYTHON`，不会覆盖 DSA 网页任务运行环境。

新增 `quant_workflow` 网页任务时，如果脚本依赖新的环境变量，必须同步评估并更新 `src/services/quant_job_service.py` 里的默认非敏感白名单，或在该 job spec 上显式声明 `env_names` / `env_prefixes`。任务中心不会把 DSA API 进程的完整环境直接透传给子进程；漏配白名单会导致脚本在网页端缺少变量，误把所有变量放开则会增加 token、密码或 LLM key 暴露到日志的风险。

`download_jqdata` 在网页端固定为 dry-run，是“验证 JQData 登录和连接是否可用”，不是全量下载。该任务可从私有 env 文件加载 `JQ_USERNAME` 和 `JQ_PASSWORD`，也只允许显式透传这两个变量；日志只记录命令和状态，不应打印或持久化真实密码。

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

从 DSA 仓库根目录运行：

```bash
PYTHONPATH=quant_workflow/scripts .venv/bin/python -m pytest quant_workflow/tests -q
```

浏览器验收必须使用当前实际监听端口，至少打开 `/`、`/data`、`/research`、`/research?mode=strategy`、`/stock-analysis`、`/portfolio` 和 `/settings`。确认六项一级导航准确、首页外部链接存在、数据页只读、两个研究模式可切换，并且页面无重叠、空白或运行时错误。

## 常见异常判断

| 现象 | 优先看哪里 |
| --- | --- |
| 页面打不开 | 核对当前 DSA 监听 PID、端口和 cwd，不永久假定 `8010` |
| 研究任务按钮失败 | `/research` 选股研究中的任务日志、`quant_workflow/reports/jobs/<job_id>/job.log` |
| 原条件重放等待超时 | 先看 DSA 日志是否已进入 Qlib 回测；这是本地计算，不等于外部网络失败，避免连续重复点击 |
| Qlib 不可训练 | `/data` 的快照身份、`/research` 的训练阻塞原因和任务日志 |
| 数据服务不可用 | `quant_data` 服务的 `/health`、`/ready` 与 DSA `/api/v1/data/status` |
| 训练效果差 | 量化研究中的回测指标和 MLflow 原始 Run |

重启 DSA 后必须同时用 `lsof -nP -iTCP:<实际端口> -sTCP:LISTEN` 和进程树确认旧
服务已经退出。关闭 screen 容器不保证其 login/Python 子进程一定随之退出；发现
孤立服务时，只能终止已经核对过身份和进程组的旧 DSA，不能按模糊进程名批量
结束，更不能影响 6006 的 MLflow。

## 阶段验收清单

- DSA 工作区无散落未跟踪源码。
- 一级导航只有首页、数据、量化研究、个股分析、持仓和设置。
- 首页保留九点猫研、MLflow 等常用入口，并以不可点击的关系图说明项目职责与数据流；不承载研究状态。
- `/data` 只读连接 `quant_data` 服务，不提供下载、发布或 Provider 构建操作。
- `/research` 同时承载选股研究和交易策略研究；`/workflow`、`/backtest` 仅作兼容路由。
- Qlib 通用训练和回测只消费 `quant_data` 已发布的不可变数据快照，并在使用前完成轻量 Provider 预检；CSI800 专用流程继续遵守自己的冻结 readiness 契约。
- MLflow 链接存在，但不替代 DSA 主 UI。
- `china-astock-quant` 的复用能力已列账，交易相关能力保持禁区。
- 同花顺只作为只读观察方向，手机连接暂不做。
- 后端 pytest、前端 test/lint/build 和浏览器 smoke 通过后，交给 Agent C 按网页步骤复测；除非明确要求，不运行 complete/OpenCode 验收。

## Agent 持续运行机制（goal 模式）

### 状态文件协议

长期 agent 每轮启动先读 `quant_workflow/reports/state/agent_state.json`，
按状态判断该做什么；每轮结束回写。不存在则初始化，损坏则本轮【阻塞】。

```json
{
  "last_data_identity": {
    "etf_snapshot_id": "cn-etf-...-2c219d19",
    "etf_content_sha256": "2f22...",
    "etf_latest_trade_date": "2026-08-07"
  },
  "last_signal_date": "2026-08-07",
  "last_weekly_roll": "2026-08-07",
  "last_monthly_check": "2026-07",
  "alerts": []
}
```

### 任务指纹与幂等

周期任务落库前查 `backtest_runs.params_json` 中是否已存在相同指纹：
`策略名 + 数据版本(snapshot_id) + 任务类型(rolling-update | data-republish) + 最新交易日`。
存在则跳过，禁止重复落库。

### 每日信号报告格式

存 `quant_workflow/reports/daily/YYYY-MM-DD.md`（信号日期 T），两个策略分开输出：

```
# 每日信号 T（执行日 = 下一交易日）

## 3ETF趋势动量（满仓/空仓）
| 标的 | 20日动量排名 | 收盘价 | MA20 | 站上均线 | 信号 | 建议仓位 |
| 159915 | 1 | ... | ... | 是 | 持有 | 100% |

## 3ETF趋势动量-广度分档（0/⅓/⅔/满仓）
同上表，仓位 = 站上 MA20 的标的数 ÷ 3
```

### 周度维护与月度漂移检测

- **周度**（幂等触发）：两个固化策略各做一次全历史滚动回测并落库
  （params 含 `rolling-update` 与数据版本）。
- **月度**（幂等触发）：计算滚动 1 年（最近 252 个交易日）的年化收益与最大回撤：
  - 一级预警：滚动 1 年年化 < 历史全部滚动 1 年年化的 P25，或滚动 1 年最大回撤
    超过历史全样本最大回撤；
  - 二级升级【策略恶化】：连续两个检测周期触发，或同一周期两个指标同时触发。
  - 预警不等于策略失效，不得据此自动修改参数或策略。
