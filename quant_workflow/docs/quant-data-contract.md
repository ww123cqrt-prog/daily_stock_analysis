# quant_data 数据治理与更新契约 v2

本文档说明 `${QUANT_DATA_ROOT}` 中每套数据是什么、如何产生、能否使用，以及更新失败时如何保护旧数据。新 Agent 不应根据目录名猜数据口径。

## 接手只做这四步

```bash
python quant_workflow/scripts/quant_data_cli.py list
python quant_workflow/scripts/quant_data_cli.py describe <dataset_id>
python quant_workflow/scripts/quant_data_cli.py resolve <dataset_id> --require-basis qfq
python quant_workflow/scripts/sync_data_manifest.py --check
```

- `list` 查看所有已登记数据。
- `describe` 查看来源、原始证券代码、价格口径、复权时间、复权算法、因子、公司行为、最新交易日和内容哈希。
- `resolve` 只返回状态和价格口径都符合要求的不可变快照，并校验文件哈希。
- `--check` 检查根清单、绝对路径、未登记目录和 README 自动表格漂移。

`legacy_unknown` 只能审计，不能通过 `resolve` 进入新训练或回测。README 与清单冲突时停止读取并修复清单，不得选择“看起来能用”的目录。

## 当前迁移状态

| 数据 | v2 状态 | 当前边界 |
|---|---|---|
| 两只 ETF raw | `audit/ready` | 腾讯同源原始价快照，可做复权证据和交叉检查 |
| 两只 ETF qfq | `canonical/ready` | 腾讯同源 raw/qfq 配对，正式可解析 |
| A 股 Baostock raw (`ashare_daily_raw`) | `canonical/ready` | 保存完整 raw 行情、来源、逐文件哈希和公司行为记录 |
| A 股 Baostock qfq (`ashare_daily_qfq`) | `canonical/ready` | 使用 Baostock `foreAdjustFactor` 本地计算，保存完整因子并用同源 qfq 核验 |
| A 股 Qlib v2 (`qlib_ashare_v2`) | `canonical/ready` | 由正式 qfq 快照生成，包含沪深300、中证500、中证800、创业板指和科创50五套指数基准，以及 CSI300/500/800 股票池，并锁定准确的上游快照 ID 和内容哈希 |
| A 股 cleaned Qlib v2 (`qlib_ashare_cleaned_v2`) | `canonical/ready` | 与 canonical 保持独立语义，保存异常屏蔽证据并复用同一套原生指数和股票池证据 |
| 旧 JQData/东方财富/AkShare 来源目录 | `audit/legacy_unknown` | 仅审计，不再直接合并生成正式输出 |
| V1 固定路径 merged/cleaned/Qlib | `legacy/ready` 兼容输出 | 仅供尚未迁移的旧程序和回滚，不是 DSA 正式输入 |
| 带 `_old` 后缀的 merged/cleaned/Qlib | `audit/legacy_unknown` | 迁移前旧数据，只用于回滚和审计 |
| PIT CSI800 和参考表 | `canonical/ready`（完整时） | 由原有事务锁和内容哈希保护，仍在逐步迁移到统一快照结构 |

首个正式 A 股 v2 快照覆盖到 `2026-07-24`。后续是否已更新到更晚日期，必须以 `quant_data_cli.py describe` 和根 `DATA_MANIFEST.json` 为准，不能把本文日期当成动态状态。迁移前旧 A 股输出改名为 `_old` 后保留，仅供回滚和审计。

## V2 内部与 V1 对外契约

V2 内部继续使用不可变快照、真实复权因子、公司行为、文件哈希和原子指针：

```text
parquet/ashare_1d/ashare_daily_raw/snapshots/<snapshot_id>/
parquet/ashare_1d/ashare_daily_qfq/snapshots/<snapshot_id>/
qlib/cn_data_ashare_v2/snapshots/<snapshot_id>/
qlib/cn_data_ashare_cleaned_v2/snapshots/<snapshot_id>/
```

尚未迁移的旧程序可以继续读取原路径：

```text
parquet/ashare_1d/merged_baostock_jqdata/
parquet/ashare_1d/merged_baostock_jqdata_cleaned/
qlib/cn_data_baostock_jqdata/
qlib/cn_data_baostock_jqdata_cleaned/
```

这些固定路径由 `publish_v2_compatibility.py` 从同一个 `ashare_daily_qfq` 快照成组生成。DSA canonical 和 cleaned 训练、预测、回测、交易日历及质量检查都不再读取它们，而是分别通过根清单解析 `qlib_ashare_v2` 和 `qlib_ashare_cleaned_v2`。兼容 Parquet 保持六位文件名、V1 列顺序和 `datetime64[ns]`；价格已经前复权，因此对外 `factor=1.0`。真实逐日因子只保存在 V2 `factors/`。兼容刷新是正式原生发布成功后的旧系统维护；它失败时记录警告，不撤销原生 V2 指针。

## 清单结构

根 `DATA_MANIFEST.json` 使用 `schema_version: 2`。`datasets` 是唯一的新接口目录，每项至少包含：

- `dataset_id`、`data_kind`、`role`、`status`；
- `source.provider`、`source.interface`、供应商证券代码；
- `price_basis`；
- 原始取得时间、复权时间、复权覆盖日期；
- 复权方法、字段、归一规则和算法版本；
- 日期范围、最新交易日、行数、证券数；
- bar/因子索引、公司行为文件及内容哈希；
- 上游快照、生成脚本和版本。

清单中的数据路径全部相对 `${QUANT_DATA_ROOT}`。`official_entries` 和 `documented_inputs` 仅为旧页面兼容，不是 v2 解析依据。

## 不可变快照

```text
<dataset_id>/
  LATEST.json
  snapshots/<snapshot_id>/
    DATASET_MANIFEST.json
    FILE_INDEX.parquet
    FACTOR_INDEX.parquet
    bars/<symbol>.parquet
    factors/<symbol>.parquet
    corporate_actions.parquet
```

- 下载和计算先写入 `staging/<run_id>/`。
- 快照通过字段、日期、主键、价格关系、复权证据和哈希校验后才发布。
- `LATEST.json` 最后原子更新，只指向完整快照。
- 失败时旧 `LATEST.json` 不变；新训练锁定启动时解析到的快照路径，不随之后的更新跳转。
- bar、因子和公司行为都有独立哈希，整个快照另有 `content_sha256`。

## 复权证据

正式 qfq/hfq 数据只接受：

1. `local_factor`：保存完整因子、本地公式和算法版本；或
2. `provider_native_paired`：保存同一供应商、同一抓取时点的完整 raw 与 qfq，并保存逐日对应因子。

只有供应商 qfq 标签、但无因子和配对 raw 的数据一律是 `legacy_unknown`。不同供应商已经复权的价格不得按日期拼接。

## ETF 更新

正式数据集：

- `etf_momentum_daily_raw`：`audit/ready/raw`；
- `etf_momentum_daily_qfq`：`canonical/ready/qfq`。

每次更新都完整下载 `SZ159915` 和 `SH513100` 的同源 raw/qfq。腾讯优先，东方财富备用，新浪旧原始价只做独立审计。发布条件：

- 每只 ETF 自己的 raw/qfq 日期完全一致；
- 两只 ETF 的最新交易日一致；
- 两只 ETF 历史日历可因停牌而不同，回测使用交集日历；
- qfq OHLC 与 raw 乘因子的关系通过校验；
- 最新有效交易日的前复权因子为 1；
- 513100 的 2022 年 1:5 拆分同时有公告和因子曲线证据；
- 两只 ETF 全部成功后才更新正式 qfq 指针。

```bash
python quant_workflow/scripts/quant_data_cli.py update \
  etf_momentum_daily_qfq --through YYYY-MM-DD
```

每日更新编排器调用同一命令，不再只刷新最近五天。

## A 股更新边界

旧下载和合并入口仍保留两项保护：

- qfq 增量更新先检查历史锚点；供应商重算历史尺度时整只股票全历史刷新；锚点无法核验时不更新该股票。
- `merge_to_qlib.py` 的 qfq 模式只能为每只股票选一个完整供应商，禁止跨供应商按日期拼接；只有 raw 模式允许整行补缺，并记录 `source_id`。

正式入口已经切换为 `ashare_daily_raw`、`ashare_daily_qfq` 和 `qlib_ashare_v2`。后续更新必须继续生成完整来源 bundle，核对 raw、`query_adjust_factor`、同源 qfq 证据和旧数据价格路径，再以不可变快照成组发布；任一证据失败时旧正式指针保持不变。其他供应商只有完成同等级 raw、因子和公司行为迁移后才能进入正式入口，禁止把不同供应商已经复权的价格直接拼接。

### A 股 v2 Baostock 事务

`download_baostock_v2.py` 使用一个受限、可恢复的来源 bundle：

- 每只股票完整下载 `adjustflag=3` raw 行情；
- 每只股票完整下载 `query_adjust_factor` 因子事件；
- 将 `foreAdjustFactor` 按事件日向后生效，生成逐日因子；
- 用 `raw OHLC × foreAdjustFactor` 重建正式 qfq，成交量和成交额不复权；
- 对最近发生因子变化的股票和固定哈希抽样股票下载供应商完整 qfq，逐日核对本地公式；
- qfq 对照只用于验证，不作为正式行情来源，也不与其他供应商 qfq 拼接。

少数新股的 raw 首日可能比第一条因子事件早 1 天。下载器只允许最多 7 个自然日的上市边界缺口，并将事件前的上市日明确记录为单位因子 1；凡发生这种补齐的股票，必须额外下载供应商完整 qfq 严格核对。超过 7 天、或核对不一致时立即停止，不能扩大补齐范围。

代码重组、重新上市等情况可能导致 `query_adjust_factor` 晚于 raw 历史多年。此时不向前套用第一条事件因子，而是下载同一 Baostock、同一抓取时点的完整 qfq，用逐日 qfq/raw OHLC 四组比值反推出缺失期因子；四组比值必须一致，推导因子变化必须写入公司行为证据。该同源配对证据不完整时，整套 A 股 v2 仍然阻止发布。

Baostock 的退市股票最后一个 `foreAdjustFactor` 不保证等于 1。此类非 1 终值不能直接拒绝，也不能自行重归一；下载器必须取得同一 Baostock 的完整 raw/qfq 配对，确认末日因子和全历史价格路径一致。正式 qfq 清单记录这类股票数量，并将复权规则写为供应商原生前复权因子加同源配对核验。

正式发布前还会对旧 `baostock_qfq` 做全量重叠审计。固定比例缩放属于可解释的前复权重归一；旧文件若因增量覆盖出现尺度拼接，只有在新 close-to-close 收益与旧文件保存的供应商 `pctChg` 对齐、而旧 close-to-close 收益不对齐时，才明确归因于旧文件。无法解释的日期缺口或价格路径差异会阻止发布，逐股票明细保存在新 qfq 快照的 `LEGACY_COMPARISON.parquet`。

下载器要求单实例、请求间隔不少于 0.2 秒、有限请求预算、最多两次尝试，并在服务端拒绝或连续失败时立即熔断。远端机器只负责生成来源 bundle；本机必须核对压缩包 SHA-256、路径安全、bundle 内文件哈希和数据契约后才能发布。

```bash
# 由受控远端 runner 生成 bundle；运行参数和远端路径写入私有运行记录
python quant_workflow/scripts/download_baostock_v2.py download \
  --universe-file stock_lists.json \
  --output-dir <staging-bundle> \
  --through YYYY-MM-DD \
  --resume

# bundle 安全传回并验证后，更新 raw/qfq 及 canonical/cleaned Qlib v2
python quant_workflow/scripts/quant_data_cli.py update \
  ashare_daily_qfq --bundle-dir <verified-bundle>
```

任意股票 raw、因子或 qfq 抽样核对失败，整个 bundle 都不能发布。完整更新链路持有 `runtime/locks/ashare_v2_update.lock` 单实例锁，第二个并发更新会在改写任何正式指针前退出。第一次发布前，命令把根清单和四个旧 `LATEST.json` 的原始内容持久化到 `runtime/transactions/ashare_v2_update.json` 并执行 `fsync`。每个 staging 快照发布时，先对全部文件和子目录执行 `fsync`，再原子 rename，并同步 staging 与正式 `snapshots/` 目录；只有这些步骤成功后才写 `LATEST.json` 和根清单。raw、qfq、canonical Qlib 和 cleaned Qlib 都通过校验后才删除该事务记录并正式提交；普通异常会立即恢复旧指针，进程被强制结束或机器断电时，下一次更新会在持有更新锁和发布锁后先自动恢复，再开始新的发布。事务记录缺失字段、内容哈希不符或无法读取时直接阻断更新，不能猜测旧状态。恢复结果写入 `runtime/status/quant_data_update_latest.json`，状态为 `rolled_back` 或 `recovered_interrupted_update`。已经完成校验但失去正式指针的不可变快照可以保留作审计，它们不会被 `resolve` 选中。默认更新到这里结束，不改四套旧固定路径。只有显式传入 `--refresh-legacy-compatibility` 才会在原生链成功后维护兼容输出；该可选步骤失败只写 `completed_with_legacy_warning`，不能撤销有效的原生快照。回滚本身失败则明确写 `rollback_failed`，并保留事务记录供下次恢复。

## Qlib 血缘

新生成的 Qlib 目录必须包含 `PROVIDER_MANIFEST.json`，记录准确的行情快照、价格口径和转换配置。训练启动后锁定该 provider；数据更新不会改变正在运行的训练输入。

```bash
python quant_workflow/scripts/build_qlib_v2.py --publish
```

该命令只解析 `ashare_daily_qfq` 的 `ready/qfq` 快照，生成不可变 `qlib_ashare_v2`，并将精确的上游快照 ID 和内容哈希写进 provider 清单。构建阶段同时写入 CSI300/500/800 股票池，训练过程不得修改快照。Qlib 指针与根清单使用同一回滚事务。

`qlib_ashare_v2` 是 DSA 的正式 Qlib 快照，包含 `SH000300`、`SH000905`、`SH000906`、`SZ399006`、`SH000688` 的 open/high/low/close/volume/factor。构建器优先复用与当前 qfq 快照、交易日历完全一致的已验证指数；不一致时只按指数代码下载和重建这五条基准，把原始 Parquet、来源清单、覆盖日期和哈希保存在 `runtime/` 的对应运行目录中，不会重下全市场行情。任一必要交易日缺失则不发布。新增单条或少量指数时使用 `update_qlib_index_benchmarks.py`，只下载明确列出的缺失指数并复用全部现有股票特征。

`qlib_ashare_cleaned_v2` 是独立的正式 cleaned 快照。它要求 canonical Qlib 与当前 qfq 上游完全一致，再记录异常扫描、屏蔽统计、证据文件及哈希；cleaned 调用方不再读取旧固定路径。

`cn_data_baostock_jqdata*` 仍是从同一 V2 快照生成的稳定兼容输出，但 DSA 主流程不依赖它。CatDesk 历史信号验证直接读取 `ashare_daily_qfq` 的 bars。迁移前 provider 位于对应 `_old` 目录，只能审计或回滚。

`run_daily_update.py` 不再构建 merged/cleaned 候选，也不再调用 `publish_daily_candidate.py`。普通 Baostock、东方财富和 AkShare 下载只更新审计缓存；只有命令行显式传入 `--ashare-v2-bundle-dir` 时，编排器才委托 `quant_data_cli.py update ashare_daily_qfq --bundle-dir ...` 事务发布正式 raw/qfq、canonical Qlib V2 和 cleaned Qlib V2。随后直接验证两个 Qlib 原生快照、Qlib 可读性、数据质量和根清单漂移。正常流程不会刷新兼容输出；兼容维护必须由命令行显式传入 `--refresh-legacy-compatibility`，网页也不接受任意 bundle 路径或该兼容刷新开关。

### 兼容输出退役边界

本轮不删除兼容输出。清单将这些输出登记为 `role: legacy` 和 `lifecycle: compatibility_output`；`status: ready` 只表示旧程序仍可读取，不代表它们是正式主入口。以后只有在生产代码不再引用旧路径、训练/预测/回测均有原生实跑证据、CatDesk 等调用方已迁移或明确保留 cleaned 需求，并获得用户单独授权后，才能提出删除计划。生产回滚只能让 `QLIB_PROVIDER_URI` 或 `QLIB_CLEANED_PROVIDER_URI` 固定到对应 `snapshots/` 下经过完整校验的原生 V2 快照；旧兼容目录只能通过显式兼容工具读取，不能重新成为 DSA 的训练、预测或回测入口。不得改写或删除原生快照和 `_old` 审计数据。

## 失败规则

- raw 下载成功、复权失败：qfq 正式指针不前进。
- 两只 ETF 只成功一只：不发布。
- 因子日期不覆盖行情、哈希变化、公司行为缺失：拒绝解析或发布。
- 请求 qfq 却解析到 raw：明确报错。
- 指数基准请求按固定间隔发送，连接错误、429 和 5xx 最多重试三次并指数退避；任一必需指数仍失败时，本轮基准快照不发布。
- 清单过期、README 漂移或出现未登记目录：`sync_data_manifest.py --check` 返回非零。
- 旧数据清理、移动或删除：必须另行 dry-run 并取得用户明确确认。

运行状态和日志不属于正式数据：状态写入 `quant_workflow/reports/`，日志写入 `quant_workflow/logs/`。账号、密码、token、券商信息、行情数据、模型和 `mlruns` 不进入 Git。
