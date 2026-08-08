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
| 统一 ETF 日线 (`cn_etf_daily`) | `canonical/ready` | 一个不可变快照同时保存腾讯同源 raw/qfq、分段复权依据和公司行为；读取方按所需价格口径解析 |
| A 股 Baostock raw (`ashare_daily_raw`) | `canonical/ready` | 保存完整 raw 行情、来源、逐文件哈希和公司行为记录 |
| A 股 Baostock qfq (`ashare_daily_qfq`) | `canonical/ready` | 使用 Baostock `foreAdjustFactor` 本地计算，保存完整因子并用同源 qfq 核验 |
| Qlib Parquet 直接适配器 | `ready` | 直接读取 `ashare_daily_qfq` 的冻结快照，只生成约几十 KB 的日历和描述符，不再复制全市场 `.bin` 行情 |
| 物理 Qlib v2 (`qlib_ashare_v2`) | 迁移期兼容 | 现有约 487 MB 快照暂留给尚未迁移的调用方、指数证据和回滚；新适配器不依赖其中的股票行情 |
| 物理 cleaned Qlib v2 (`qlib_ashare_cleaned_v2`) | 迁移期兼容 | cleaned 语义尚未改成直接读取路径，不能在下游切换完成前删除或冒充已迁移 |
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
runtime/qlib_direct/providers/ashare_daily_qfq/<snapshot_id>/<membership_snapshot_hash>/
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

ETF 只有一个正式数据集：`cn_etf_daily`。一个不可变快照内部同时保存：

```text
cn_etf_daily/
  LATEST.json
  snapshots/<snapshot_id>/
    raw/
    qfq/
    factors/
    corporate_actions.parquet
    DATASET_MANIFEST.json
```

raw、qfq和复权因子是同一份正式数据的配套组成部分，不是三个可独立发布的
数据集。`LATEST.json` 只指向整包校验通过的快照。发布事务失败时不切换；发布
成功后，新快照已经成为正式数据，随后才清理过旧快照并再次同步根清单。正常情况
只保留当前和上一次两份；若这项发布后维护失败，返回
`published_with_warnings` 并保留已发布指针，不能误报为“没有发布”。

当前正式证券池为51只，即已验证的原50只与本地 `513100` 的并集。普通日期更新
冻结当前 `ETF_UNIVERSE.parquet`，不再根据新浪成交额重新选择证券，也不把证券池
变化混入日期更新。证券池包含 `512890`、`513010`、`510720`、`159915` 和
`513100`。
其中 `513010` 的当前名称是恒生科技ETF易方达；红利国企ETF国泰的代码是
`510720`，两者均保留以避免代码和名称误配。普通更新只请求每只证券的有限重叠
窗口和缺失日期；目标日期已存在时零网络请求。重叠数据证明某只证券的复权历史
变化时，只允许单独修复该证券。下载器使用全局限速、有限重试和单实例锁；任一
证券缺少同源 raw/qfq 或因子验证失败时，本轮不发布。
腾讯 ETF 前复权不总是纯乘法：拆分使用乘数，现金分配还可能产生价格偏移。
因子证据因此按连续区间保存 `qfq_ohlc = raw_ohlc * factor + offset`，同一区间只
使用一组参数，最新交易日必须归一为 `factor=1`、`offset=0`。每日单独反推产生的
舍入抖动不能登记成公司行为。

```bash
python quant_workflow/scripts/quant_data_cli.py update \
  cn_etf_daily --through YYYY-MM-DD

python quant_workflow/scripts/quant_data_cli.py \
  validate cn_etf_daily --release
```

第二条命令是ETF正式发布的单命令验收：只读、无网络，只检查当前ETF快照、允许
保留的一份备份、raw/qfq/factors、公司行为、证券池、根清单和旧ETF入口，不扫描无关A股
数据。每日更新编排器会在ETF更新后、报告生成前自动调用；成功状态固定为
`full_validation_passed`，失败会给出具体项目并返回非零。

全量重建不是普通更新的兜底。只有当前正式快照经全检确认严重损坏、唯一备份经过
实际验证仍无法恢复，并且用户明确批准本次全量重建时才允许执行；任何条件不明确
都必须停止并询问用户。完整下载脚本还会在联网前检查对应的显式确认参数。

读取方使用同一个数据集ID并声明所需口径：

```bash
python quant_workflow/scripts/quant_data_cli.py resolve cn_etf_daily --require-basis raw
python quant_workflow/scripts/quant_data_cli.py resolve cn_etf_daily --require-basis qfq
```

项目内ETF与校验操作由四个协作技能约束，单一真源位于 `.claude/skills/`：

- `quant-data-etf-full-download`
- `quant-data-etf-update`
- `quant-data-sample-validation`
- `quant-data-full-validation`

抽检只检查确定性样本并报告剩余文件；正式发布验收必须使用全检。Codex等外部
Agent通过指向上述目录的本地技能链接发现它们，不维护第二份同义内容。

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

## Qlib 直接适配层

Qlib 的模型、Alpha158 表达式、训练、策略和回测逻辑不变。`quant_data` 只替换 Qlib 最底层的三类数据读取：

- `QuantDataCalendarProvider`：读取冻结快照派生的交易日历；
- `QuantDataInstrumentProvider`：将 `all` 和历史时点 CSI300/500/800 区间映射给 Qlib；
- `QuantDataFeatureProvider`：按股票直接读取 `bars/<symbol>.parquet` 并映射 OHLC、成交量、成交额、换手率、涨跌幅、交易状态、ST、VWAP 和单位 `$factor`。

```bash
python quant_workflow/scripts/build_qlib_direct_provider.py
```

该命令解析当前 `ashare_daily_qfq`，写入：

```text
runtime/qlib_direct/providers/ashare_daily_qfq/<snapshot_id>/<membership_snapshot_hash>/
  CALENDAR.parquet
  MEMBERSHIP_INTERVALS.parquet
  PROVIDER_DESCRIPTOR.json
```

这里没有 `features/`，也没有 `.bin` 行情。当前正式实测目录约 92 KB；作为对照，保留的单份物理 Qlib v2 股票快照约 487 MB。前复权价格已经在 `ashare_daily_qfq` 中完成，因此 Provider 对 Qlib 暴露的 `$factor` 在有行情的日期为 `1.0`，不会再次复权；无行情日期保持 `NaN`，不会向前填充。

描述符记录行情快照 ID、内容哈希、文件索引哈希、股票池快照哈希和适配器版本，并把股票池区间复制为一份很小的冻结证据。一个训练进程启动后只读该描述符锁定的行情与股票池，即使 `LATEST.json` 或正式股票池随后更新，也不会切到新数据。每只股票第一次读取时会核对其 Parquet 哈希。已有描述符再次启动只做快速身份核对；完整 5516 只股票验证只发生在新行情快照第一次生成描述符时。

真实验证已覆盖基础字段、滚动表达式、两进程读取和 Alpha158 的 158 个因子。对 3 只股票、8 个字段/表达式的对照结果与同源旧 `.bin` 完全一致。

当前迁移边界必须说清：直接适配器已经可用，但现有 canonical/cleaned 训练入口尚未全部改用它；独立的五指数正式 Parquet 数据集和 cleaned 直接语义也仍需接线。因此旧 `qlib_ashare_v2`、`qlib_ashare_cleaned_v2` 暂时只保留作迁移兼容、指数证据和回滚，不能现在删除。后续调用方应使用 `_paths.init_qlib_direct()`，而不是把描述符目录交给普通 `qlib.init(provider_uri=...)`；后者不会自动加载自定义 Provider。

### 兼容输出退役边界

本轮不删除兼容输出。清单将这些输出登记为 `role: legacy` 和 `lifecycle: compatibility_output`；`status: ready` 只表示旧程序仍可读取，不代表它们是正式主入口。以后只有在生产代码不再引用旧路径、训练/预测/回测均有原生实跑证据、CatDesk 等调用方已迁移或明确保留 cleaned 需求，并获得用户单独授权后，才能提出删除计划。生产回滚只能让 `QLIB_PROVIDER_URI` 或 `QLIB_CLEANED_PROVIDER_URI` 固定到对应 `snapshots/` 下经过完整校验的原生 V2 快照；旧兼容目录只能通过显式兼容工具读取，不能重新成为 DSA 的训练、预测或回测入口。不得改写或删除原生快照和 `_old` 审计数据。

## 失败规则

- raw 下载成功、复权失败：qfq 正式指针不前进。
- 统一证券池中任一 ETF 缺少同源 raw/qfq 或复权证据：不发布。
- 因子日期不覆盖行情、哈希变化、公司行为缺失：拒绝解析或发布。
- 请求 qfq 却解析到 raw：明确报错。
- 指数基准请求按固定间隔发送，连接错误、429 和 5xx 最多重试三次并指数退避；任一必需指数仍失败时，本轮基准快照不发布。
- 清单过期、README 漂移或出现未登记目录：`sync_data_manifest.py --check` 返回非零。
- 发布后保留策略只允许删除 `cn_etf_daily/snapshots/` 内早于唯一回滚版本的普通
  快照目录；清理失败记为 `published_with_warnings`，不回滚已经成功的发布。
- 保留策略之外的旧数据清理、移动或删除：必须另行 dry-run 并取得用户明确确认。

运行状态和日志不属于正式数据：状态写入 `quant_workflow/reports/`，日志写入 `quant_workflow/logs/`。账号、密码、token、券商信息、行情数据、模型和 `mlruns` 不进入 Git。
