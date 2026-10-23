# 脚本策略资产与回测运行历史

本文档描述 `/backtest` 页面「脚本交易策略」Tab 的策略资产保存与回测结果持久化能力。

## 定位

量化研究工作台第一阶段（Python 策略 → 回测 → 结果）的最小资产闭环：
- **策略资产（strategy_assets）**：命名的、可复用的 Python 交易策略。保存前必须通过与执行时相同的静态安全检查（`src/core/script_strategy.validate_strategy_script`），并记录脚本 SHA256 与脚本 API 版本。
- **回测运行（backtest_runs）**：每次脚本回测的持久化记录。无论策略是否已保存都会落库；包含数据快照身份、参数、指标与完整报告，可在历史列表中回看与对比两次运行。

这是「Strategy / Backtest 资产」在资产流水线（研究 → 信号 → 组合 → 策略 → 回测 → 验证 → 结论）上的最小落地。后续条件策略 Builder 与选股模型组合都会复用这套「策略保存 + 运行落库」基础设施。

## 股票池（universe）

新版默认接入支持以下股票池；显式传入旧 `quant_data_root` 的历史复现仍使用原快照读取器：

- **ETF 池**：`cn_etf_daily` 正式发布的 ETF 列表，使用 qfq 口径；
- **CSI800**：`ashare_daily_raw` 模块的 qfq 组件和 `pit_csi800/daily` 历史成分；
- **自定义列表**：用户手输 A 股或 ETF，按所属模块读取；A 股池另读创业板指；
- **全 A**：当前只读服务的 `security_master` 查询上限低于完整主表行数，默认入口会明确拒绝，不缩小股票池冒充全 A 回测。

数据机制：

- 默认读取本机 `quant_data` 只读服务的 active `production.v2`。回测必须给出起止日期，先检查 `/ready`，再按固定发布代次、release-set SHA 和模块内容身份请求严格覆盖的行情；返回 409 时停止回测。输入矩阵写入 DSA 的 `quant_workflow/reports/cache/universe/` 供沙箱执行，每次运行都重新向服务核对覆盖情况，不复用旧矩阵跳过数据门槛。
- 起始日是本次输入矩阵的第一天，脚本需要的历史窗口也从这一天积累；跨版本比较必须显式使用相同预热区间。
- 沙箱脚本通过矩阵视图访问数据（`close/open/volume/history/returns/sma/rank_by_return/universe`），`rank_by_return` 只在当日成员掩码内排名；正式交易资格仍须另行核对停牌、ST 等状态。
- 新版服务不会把覆盖不足的区间自动截短成成功结果。旧版显式快照复现仍保留原有截断语义。
- 时间预算分层：脚本逻辑护栏 30 秒（防死循环）；数据加载与回测执行不在护栏内；前端请求长超时（10 分钟）。

冻结的 TQM-20 三 ETF 区间结果可用 `python quant_workflow/scripts/compare_production_v2_tqm20.py` 对照。脚本先按旧报告记录的快照身份复跑控制组，再读取正式服务的严格 qfq 查询；沿用旧执行器和旧费用口径，写入新的 `reports/experiments/tqm20_production_v2_*` 目录。查询证明不足时只写阻断原因，不生成新版收益率。

## 存储

- 默认数据库文件：`quant_workflow/reports/strategy_assets.db`（SQLite，WAL 模式）。
- 可用环境变量 `DSA_STRATEGY_ASSETS_DB` 覆盖路径。
- 与 `quant_research.db`（研究 CRUD 初稿）相互独立；两者在工作台资产模型定稿后再统一收敛。

### 表结构

`strategy_assets`：`id / name / description / script / script_sha256 / api_version / archived / created_at / updated_at`

`archived` 默认为 `false`，旧数据库自动补列，已有脚本与回测不重写。归档和恢复只修改策略元数据，保留资产 ID、脚本哈希以及所有 `backtest_runs.strategy_id` 关联。

`backtest_runs`：`id / strategy_id（可空，外键 ON DELETE SET NULL）/ strategy_name / script_sha256 / symbols_json / params_json / data_quality_json / metrics_json / report_json / status / error_message / created_at`

列表查询刻意不读取 `report_json`（净值曲线与逐笔成交体积较大），详情接口才返回完整报告。

## API

全部挂载在 `/api/v1/backtest` 前缀下：

- `GET /strategies`：主策略列表（`archived=false`，不含脚本正文）
- `GET /strategies?archived=true`：历史归档列表（不含脚本正文）
- `POST /strategies`：保存策略（静态检查失败返回 400）
- `GET /strategies/{id}`：策略详情（含脚本）
- `PATCH /strategies/{id}`：更新名称/描述/脚本；`{"archived": true}` 归档，`{"archived": false}` 恢复到主列表（脚本变更重新校验并更新哈希）
- `DELETE /strategies/{id}`：删除（204；历史运行保留并解除关联）
- `GET /runs?limit=&offset=&strategy_id=&script_sha256=`：运行历史摘要
- `GET /runs/{id}`：运行详情（含完整回测报告）

脚本回测入口 `POST /backtest/strategy/script` 新增可选字段 `strategy_asset_id`；回测完成后自动落库，并在响应中返回 `run_id`。落库为 best-effort：失败只记录警告，绝不影响回测结果。

## 安全边界（不变）

- 保存与执行使用同一静态检查：禁 `import/open/eval/exec/class/async/with` 等，脚本上限 50KB / 5000 AST 节点。
- 执行仍在 macOS `sandbox-exec` 隔离子进程中进行（禁网络、禁写文件，8 秒超时）。
- 策略资产仅存储脚本文本与哈希，不存储任何运行时凭证。

## 前端

`ScriptStrategyBacktestPanel` 编辑器上方分为「主策略」「历史归档」「脚本模板」，默认只显示主策略。已保存资产可归档或恢复，页面不再用删除按钮整理策略。归档资产仍可加载脚本和选入「策略对比」，对比项标注「历史」。详情与历史回测接口均可继续按原 ID 读取。

右侧提供 `StrategyRunHistory` 历史运行列表（时间、状态、总收益、最大回撤、数据快照日期），点击任意记录即可加载完整报告进行对比。

2026-10-18 整理后的主列表为 **BVR-20 趋势量能状态切换 V0.1** 和 **TQM-20 趋势质量动量 V1.6**。另外五条已保存的动量策略归档，原版本目录和研究报告保留原路径。这里的主列表归属不改变策略规则、历史回测结果或影子验证要求。

## 验证

- 后端：`tests/test_strategy_assets_repo.py`、`tests/test_backtest_assets_api.py`
- 前端：`apps/dsa-web/src/api/__tests__/backtest.test.ts`

## 固化策略与实验隔离（goal 模式）

### 固化策略清单（只读）

`strategy_assets` 表中以下策略为**已定版固化策略**，脚本视为只读资产：

- 「TQM-20趋势质量动量」= **V1.6 正式版**（2026-08-17 定稿，机制命名 Trend Quality Momentum 20）：
  - Universe：159915 / 513100 / 512890 + 现金
  - Signal：20日收益 m × 路径效率² 陡惩罚（eff=|m|/Σ|日收益|，score = m − (1−eff²)×|m|）
  - Filter：20 日年化波动率 < 40%
  - Exit：持仓从 40 日高点回撤 ≥ 12% 清仓
  - Execution：same_day_close（T 收盘基准，与 V1.4 生产一致，依托 2026-07-06 盘后固定价格交易；
    标准引擎原生口径为 T+1 开盘，复跑结果为 47.58% / 1.68 / −19.21%（451 笔），
    两口径差异已在 R083/R085 审计中如实记录）；万 0.6；lot 100；Position 100%/0%
  - 基准指标（run `caeef927`）：CAGR 46.7% / Sharpe 1.87 / MDD −18.8% / Calmar 2.49
  - 验证：三窗口、熊市零代价、跨4池、Bootstrap、换手/费用、时间冻结 4/4、分年零负（R083-R085）
  - 定位：历史稳健性高，不承诺未来收益；影子验证后方可实盘
  - 前身（中间产物，不维护）：合并B / G2 门控 / 趋势效率动量

### 已登记正式资产但仍需影子验证

- 「BVR-20趋势量能状态切换」= **V0.1 正式资产/影子候选**（资产 ID `50ad6eda-1a4e-4765-8b89-152f193c4bb1`；对应 BBOLL 研究线 V3.4 `BStrongConfirm3`，2026-08-14 数据截止）：
  - 名称：Bollinger Volume Regime 20（20日布林量能状态切换策略）
  - 核心：BOLL、成交量和创业板指 MA20 牛熊切换；弱势模式放量超跌分批建仓，强势模式回调后确认买入；V3.4 额外要求创业板指连续 3 日收盘不低于 MA20
  - DSA 数据接口：`data.volume()`、`data.history(symbol, count, "volume")`、`data.index_close()`、`data.index_history()`；旧 bundle 缺少这些数组时会回退为空值，避免破坏旧资产读取
  - 定位：已登记到 `strategy_assets.db`，可由 DSA 沙箱读取和执行；尚无对应正式 `backtest_runs`，且未通过随机股票池和滚动样本外验证，不得直接实盘
  - 正式资产登记不等于实盘批准；影子验证是进入实盘白名单的前置条件

### 历史归档（仅作对照）

- 「3ETF趋势动量」= **V1.4 历史冻结基准**（2026-10-18 归档；原 2026-08-15 冻结，2026-08-17 增加可选 min_hold）：
  - Universe：159915 / 513100 / 512890 + 现金
  - Signal：20 日动量排名
  - Filter：20 日年化波动率 < 40%
  - Exit：持仓标的从 40 日高点回撤 ≥ 12% 清仓（不受最短持有期锁定）
  - Re-entry：立即（清仓后下一交易日重新评估选股条件）
  - Market Filter：无；Position：100% / 0%
  - Execution：same_day_close（T 收盘成交，2026-08-15 起新基准）
  - **可选参数 `context.min_hold` 最短持有期（交易日）：1=默认（无额外锁定，与收盘执行框架天然一致）；
    2=持有≥2天才允许动量换仓；3=持有≥3天。锁定仅约束动量换仓（round_078）**
  - 基准指标（min_hold=1，T 收盘）：CAGR 41.1% / Sharpe 1.64 / MDD −18.8% / Calmar 2.19（run `6ddf680c`）；
    H2：CAGR 41.1% / MDD −20.3% / 换手 −11%（run `bc161a7c`）；H3：CAGR 38.7% / MDD −17.2% / Calmar 2.29 / 换手 −20%（run `ac58b9e9`）
  - 阈值 12% 为弱确认参数（研究期不可区分，2022 验证期单调确认）；40 日窗口时序验证通过
  - 排名缓冲（Top2/Top3）实验结论：有害，不采用（round_078）
- 「3ETF趋势动量-广度分档」：历史变体（V1.1 时代仓位实验），非固化版。
- 「3ETF趋势动量-缓冲稳定版」「动量轮动-周频L20N2」和「123」：保留原脚本及已有回测作历史对照；周频版和预置脚本不作为 TQM-20 的直接前身。

非用户明确要求，禁止对这些资产执行更新或删除；实验必须复制脚本到
`quant_workflow/reports/experiments/` 或 `/tmp` 后再修改。

### 落库规范（params 约定）

正式结果落库时 `params_json` 必须携带以下字段，供幂等检查与前端核对：

- `note`：`rolling-update`（周度滚动）或 `data-republish`（数据重发布重算）；
- `data_version`：当前 ETF 快照 `snapshot_id`；
- `start_date` / `end_date`：回测区间；
- 实验性回测**不落库**；若实验必须产生 run 供讨论，存 `reports/experiments/`，
  不得写入 `backtest_runs` 表。

### 历史运行表卫生

`backtest_runs` 是用户的正式记录，禁止写入实验中间产物；同一任务指纹
（策略名 + 数据版本 + 任务类型 + 最新交易日）只允许存在一条 run。
