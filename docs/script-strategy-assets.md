# 脚本策略资产与回测运行历史

本文档描述 `/backtest` 页面「脚本交易策略」Tab 的策略资产保存与回测结果持久化能力。

## 定位

量化研究工作台第一阶段（Python 策略 → 回测 → 结果）的最小资产闭环：
- **策略资产（strategy_assets）**：命名的、可复用的 Python 交易策略。保存前必须通过与执行时相同的静态安全检查（`src/core/script_strategy.validate_strategy_script`），并记录脚本 SHA256 与脚本 API 版本。
- **回测运行（backtest_runs）**：每次脚本回测的持久化记录。无论策略是否已保存都会落库；包含数据快照身份、参数、指标与完整报告，可在历史列表中回看与对比两次运行。

这是「Strategy / Backtest 资产」在资产流水线（研究 → 信号 → 组合 → 策略 → 回测 → 验证 → 结论）上的最小落地。后续条件策略 Builder 与选股模型组合都会复用这套「策略保存 + 运行落库」基础设施。

## 股票池（universe）

脚本策略回测支持四个股票池：

- **ETF 池**：`cn_etf_daily` 全部 51 只 ETF（qfq 口径）；
- **CSI800**：`ashare_daily_qfq` + `pit_csi800` 历史时点成员——每日可交易成员随指数成分调整变化（无幸存者偏差）；
- **全 A**：`ashare_daily_qfq` + `robust_reference` 上市/退市主表——每日可用集按上市日/退市日时点过滤；
- **自定义列表**：用户手输标的（从 A 股快照读取）。

数据机制：

- 全池数据由 DSA 主进程按 quant_data 不可变快照身份（snapshot_id + manifest_sha256）组装为向量化矩阵（date × symbol），缓存于 `quant_workflow/reports/cache/universe/<快照键>/`；同一快照缓存永久有效，首次构建后再次回测仅需秒级加载。
- 沙箱脚本通过矩阵视图访问数据（`close/open/history/returns/sma/rank_by_return/universe`），`rank_by_return` 只在当日可交易成员内排名，停牌/未上市/已退市标的不可见。
- 回测区间自动截断到池成员数据最后有效日（CSI800 成员数据落后行情时会给出明确 warning）。
- 时间预算分层：脚本逻辑护栏 30 秒（防死循环）；数据加载与回测执行不在护栏内；前端请求长超时（10 分钟）。

## 存储

- 默认数据库文件：`quant_workflow/reports/strategy_assets.db`（SQLite，WAL 模式）。
- 可用环境变量 `DSA_STRATEGY_ASSETS_DB` 覆盖路径。
- 与 `quant_research.db`（研究 CRUD 初稿）相互独立；两者在工作台资产模型定稿后再统一收敛。

### 表结构

`strategy_assets`：`id / name / description / script / script_sha256 / api_version / created_at / updated_at`

`backtest_runs`：`id / strategy_id（可空，外键 ON DELETE SET NULL）/ strategy_name / script_sha256 / symbols_json / params_json / data_quality_json / metrics_json / report_json / status / error_message / created_at`

列表查询刻意不读取 `report_json`（净值曲线与逐笔成交体积较大），详情接口才返回完整报告。

## API

全部挂载在 `/api/v1/backtest` 前缀下：

- `GET /strategies`：策略列表（不含脚本正文）
- `POST /strategies`：保存策略（静态检查失败返回 400）
- `GET /strategies/{id}`：策略详情（含脚本）
- `PATCH /strategies/{id}`：更新名称/描述/脚本（脚本变更重新校验并更新哈希）
- `DELETE /strategies/{id}`：删除（204；历史运行保留并解除关联）
- `GET /runs?limit=&offset=&strategy_id=&script_sha256=`：运行历史摘要
- `GET /runs/{id}`：运行详情（含完整回测报告）

脚本回测入口 `POST /backtest/strategy/script` 新增可选字段 `strategy_asset_id`；回测完成后自动落库，并在响应中返回 `run_id`。落库为 best-effort：失败只记录警告，绝不影响回测结果。

## 安全边界（不变）

- 保存与执行使用同一静态检查：禁 `import/open/eval/exec/class/async/with` 等，脚本上限 50KB / 5000 AST 节点。
- 执行仍在 macOS `sandbox-exec` 隔离子进程中进行（禁网络、禁写文件，8 秒超时）。
- 策略资产仅存储脚本文本与哈希，不存储任何运行时凭证。

## 前端

`ScriptStrategyBacktestPanel` 编辑器上方提供策略资产选择/保存/删除，右侧提供 `StrategyRunHistory` 历史运行列表（时间、状态、总收益、最大回撤、数据快照日期），点击任意记录即可加载完整报告进行对比。

## 验证

- 后端：`tests/test_strategy_assets_repo.py`、`tests/test_backtest_assets_api.py`
- 前端：`apps/dsa-web/src/api/__tests__/backtest.test.ts`

## 固化策略与实验隔离（goal 模式）

### 固化策略清单（只读）

`strategy_assets` 表中以下策略为**已定版固化策略**，脚本视为只读资产：

- 「3ETF趋势动量」：满仓/空仓两档（20 日动量 Top1 + 收盘价 > MA20）。
- 「3ETF趋势动量-广度分档」：仓位 = 站上 MA20 的标的数 ÷ 3（0/⅓/⅔/满仓）。

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
