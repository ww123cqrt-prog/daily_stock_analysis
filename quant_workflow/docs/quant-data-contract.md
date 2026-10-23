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
| Qlib Parquet 直接适配器 | `canonical/ready` | 正式运行入口，直接读取 `ashare_daily_qfq` 和 `cn_index_daily` 的冻结快照，只生成约几十 KB 的日历、股票池副本和描述符，不复制全市场 `.bin` 行情 |
| cleaned Qlib 直接适配器 | `canonical/ready` | 与 canonical 使用同一份 qfq Parquet，通过冻结清洗计划在读取时屏蔽异常，不生成第二份行情数据 |
| 物理 Qlib v2 (`qlib_ashare_v2`) | `legacy/audit` | 旧 `.bin` 快照保留作兼容、审计和回滚；不再是训练、预测、回测或质量报告的默认输入 |
| 物理 cleaned Qlib v2 (`qlib_ashare_cleaned_v2`) | `legacy/audit` | 旧 cleaned `.bin` 快照保留作兼容、审计和回滚；不再是正式 cleaned Provider |
| 旧 JQData/东方财富/AkShare 来源目录 | `audit/legacy_unknown` | 仅审计，不再直接合并生成正式输出 |
| V1 固定路径 merged/cleaned/Qlib | `legacy/ready` 兼容输出 | 仅供尚未迁移的旧程序和回滚，不是 DSA 正式输入 |
| 带 `_old` 后缀的 merged/cleaned/Qlib | `audit/legacy_unknown` | 迁移前旧数据，只用于回滚和审计 |
| PIT CSI800 (`pit_csi800`) | `canonical/ready` | `LATEST.json` 指向不可变快照；提供每日成员、成员区间和来源清单组件 |
| 鲁棒参考表 (`robust_reference`) | `canonical/ready` | 冻结对应 PIT CSI800 快照；提供证券主表、行业区间、ST、停牌和来源清单组件 |
| PIT 基本面 (`ashare_fundamentals_pit`) | `canonical/ready` | 东方财富历史报表，使用公告/更新时间后一个交易日生效，来源覆盖率不低于 98% |
| 分钟线、盘中快照 | 尚未建立 | 日频读取器和 Qlib Provider 不会推导或伪造分钟数据 |
| 全球日线 | `audit/legacy_unknown` | 尚无可复现 canonical 快照，不能进入正式港美日韩训练或回测 |

首个正式 A 股 v2 快照覆盖到 `2026-07-24`。后续是否已更新到更晚日期，必须以 `quant_data_cli.py describe` 和根 `DATA_MANIFEST.json` 为准，不能把本文日期当成动态状态。迁移前旧 A 股输出改名为 `_old` 后保留，仅供回滚和审计。

## V2 内部与 V1 对外契约

V2 内部继续使用不可变快照、真实复权因子、公司行为、文件哈希和原子指针：

```text
parquet/ashare_1d/ashare_daily_raw/snapshots/<snapshot_id>/
parquet/ashare_1d/ashare_daily_qfq/snapshots/<snapshot_id>/
runtime/qlib_direct/providers/ashare_daily_qfq/<snapshot_id>/<membership_snapshot_hash>/
```

需要回放旧格式的兼容工具可以继续读取原路径，但正式 DSA 运行不读取这些路径：

```text
parquet/ashare_1d/merged_baostock_jqdata/
parquet/ashare_1d/merged_baostock_jqdata_cleaned/
qlib/cn_data_baostock_jqdata/
qlib/cn_data_baostock_jqdata_cleaned/
```

这些固定路径由 `publish_v2_compatibility.py` 从同一个 `ashare_daily_qfq` 快照成组生成，只服务仍需旧格式的显式兼容工具。DSA canonical 和 cleaned 训练、预测、回测、交易日历及质量检查都直接使用 `runtime/qlib_direct/providers/.../PROVIDER_DESCRIPTOR.json`。兼容 Parquet 保持六位文件名、V1 列顺序和 `datetime64[ns]`；价格已经前复权，因此对外 `factor=1.0`。真实逐日因子只保存在 A 股正式 qfq 快照。兼容刷新不是正式数据发布的一部分。

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

### 非行情组件与统一读取接口

PIT 股票池、参考表和基本面数据同样使用不可变快照，但以
`ARTIFACT_INDEX.parquet` 登记文件，以 `components` 声明可读取的逻辑组件。读取方
只能按组件名解析，不能传入任意路径：

```bash
python quant_workflow/scripts/quant_data_cli.py resolve \
  pit_csi800 --component daily
python quant_workflow/scripts/quant_data_cli.py resolve \
  robust_reference --component security_master
python quant_workflow/scripts/quant_data_cli.py resolve \
  ashare_fundamentals_pit --component daily
```

A 股、ETF 和指数日线共用一个只读接口。它按正式文件索引解析证券代码，验证所选
Parquet 哈希，再做字段和日期裁剪：

```bash
python quant_workflow/scripts/quant_data_cli.py bars \
  ashare_daily_qfq SH600000 --basis qfq \
  --fields close volume --start 2026-08-01 --end 2026-08-07

python quant_workflow/scripts/quant_data_cli.py bars \
  cn_etf_daily 513100 --basis qfq --fields close
```

项目 Python 调用方使用：

```python
from quant_data_reader import read_bars

bars = read_bars(
    "cn_index_daily",
    "SH000300",
    quant_data_root=quant_data_root,
    fields=["close", "volume"],
    start="2026-08-01",
    end="2026-08-07",
)
```

A 股增量下载的证券清单必须从本次锁定的 base snapshot 导出，不能直接复用独立的
`stock_lists.json`：

```bash
python quant_workflow/scripts/quant_data_cli.py export-universe \
  ashare_daily_qfq --basis qfq --output <run-universe.json>
```

导出文件记录 `snapshot_id`、`manifest_sha256`、`latest_trade_date` 和
`symbol_count`，用于在远端下载前确认证券集合与正式输入一致。

该接口当前契约是**日频 bar**。分钟线和盘中快照需要独立 canonical 数据集、索引
和读取契约；在这些数据建立前，打板等盘中研究保持暂停。全球日线当前也仍是
`audit/legacy_unknown`，不能通过此接口绕过治理状态进入正式训练。

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
默认保留最近3份完整不可变快照；若这项发布后维护失败，返回
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

正式数据入口是 `ashare_daily_raw`、`ashare_daily_qfq`、`cn_index_daily` 和 direct Qlib Provider 描述符。后续更新必须继续生成完整来源 bundle，核对 raw、`query_adjust_factor`、同源 qfq 证据和旧数据价格路径，再以不可变快照成组发布；任一证据失败时旧正式指针保持不变。其他供应商只有完成同等级 raw、因子和公司行为迁移后才能进入正式入口，禁止把不同供应商已经复权的价格直接拼接。每日自动更新当前按用户要求暂停，恢复前不得启动下载或切换 Provider。

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

CLI 必须先读取 `BUNDLE_MANIFEST.json.bundle_mode`。`incremental` bundle 只调用
`incremental_update_ashare_v2.py`：复用当前完整历史，合并有限重叠窗口，只有完整
复权因子指纹变化的证券才重算自身全历史 qfq；`full` bundle 才调用完整构建器。
两种模式都将 raw/qfq 成组校验并原子发布，禁止把增量 bundle 误送到完整构建器。

增量发布后使用下面两条只读命令验收。省略 `--candidate-snapshot` 时，候选固定为
命令启动时解析到的当前正式快照；`--output` 可将与标准输出相同的 JSON 结果原子
写入报告文件。

```bash
# 逐一读取旧、新快照的全部证券，证明新快照只是旧快照的合规增量
python quant_workflow/scripts/quant_data_cli.py \
  --root /Users/cq/WorkingProjects/quant_data \
  compare-snapshots ashare_daily_raw \
  --base-snapshot <old-raw-snapshot-id> \
  --candidate-snapshot <new-raw-snapshot-id> \
  --output <raw-comparison.json>

# 从新增区间确定性抽样，并逐行对照发布时保留的本地 Baostock bundle
python quant_workflow/scripts/quant_data_cli.py \
  --root /Users/cq/WorkingProjects/quant_data \
  sample-increment ashare_daily_raw \
  --base-snapshot <old-raw-snapshot-id> \
  --candidate-snapshot <new-raw-snapshot-id> \
  --bundle-dir <verified-local-bundle> \
  --sample-size 100 \
  --comparison-report <raw-comparison.json> \
  --output <raw-sample.json>
```

`compare-snapshots` 对 raw 要求旧截止日及以前逐行不变；对 qfq 只允许因逐日复权
因子数值确实变化而重算旧价格。清单中的 `factor_changed_symbols` 按完整公司行为事件
历史核对；报告另行列出因子数值变化数、因子证据元数据变化数和真正改写旧 qfq
价格的证券数，不能把三种口径混为一个数字。两种价格口径都检查证券集合、文件
哈希、日期顺序、重复日期、新增日期边界、OHLC 与成交量。
`sample-increment` 不联网，raw 直接对照 bundle 原始行情，qfq 则使用 bundle 的 raw
与完整因子证据按冻结公式重建后对照。抽检报告必须列出已检查和未检查证券数量；
命令会在当前进程内重新执行完整增量对比，并要求 `--comparison-report` 与重算结果
完全一致，不能用人工构造或过期报告跳过历史检查。qfq 的增量基础身份通过旧 qfq
快照声明的唯一 raw 上游绑定，和候选清单的 `source.base_snapshot_id` 对照。抽检通过
不能替代完整增量对比。

任意股票 raw、因子或 qfq 抽样核对失败，整个 bundle 都不能发布。完整更新链路持有 `runtime/locks/ashare_v2_update.lock` 单实例锁，第二个并发更新会在改写任何正式指针前退出。第一次发布前，命令把根清单以及 raw、qfq 两个正式 `LATEST.json` 的原始内容持久化到 `runtime/transactions/ashare_v2_update.json` 并执行 `fsync`。每个 staging 快照发布时，先对全部文件和子目录执行 `fsync`，再原子 rename，并同步 staging 与正式 `snapshots/` 目录；只有这些步骤成功后才写 `LATEST.json` 和根清单。raw、qfq 发布完成后，事务继续生成并校验 canonical、cleaned 两个 metadata-only Direct Provider 描述符；两者都成功才删除事务记录并正式提交。普通异常会立即恢复旧指针，进程被强制结束或机器断电时，下一次更新会在持有更新锁和发布锁后先自动恢复，再开始新的发布。事务记录缺失字段、内容哈希不符或无法读取时直接阻断更新，不能猜测旧状态。恢复结果写入 `runtime/status/quant_data_update_latest.json`，状态为 `rolled_back` 或 `recovered_interrupted_update`。已经完成校验但失去正式指针的不可变快照可以保留作审计，它们不会被 `resolve` 选中。默认更新到这里结束，不改旧固定路径或物理 `.bin` 数据。只有显式传入 `--refresh-legacy-compatibility` 才会在正式链成功后维护兼容输出；该可选步骤失败只写 `completed_with_legacy_warning`，不能撤销有效的原生快照。回滚本身失败则明确写 `rollback_failed`，并保留事务记录供下次恢复。

## Qlib 直接适配层

Qlib 的模型、Alpha158 表达式、训练、策略和回测逻辑不变。`quant_data` 只替换 Qlib 最底层的三类数据读取：

- `QuantDataCalendarProvider`：读取冻结快照派生的交易日历；
- `QuantDataInstrumentProvider`：将 `all` 和历史时点 CSI300/500/800 区间映射给 Qlib；
- `QuantDataFeatureProvider`：按股票直接读取 `bars/<symbol>.parquet` 并映射 OHLC、成交量、成交额、换手率、涨跌幅、交易状态、ST、VWAP 和单位 `$factor`。

```bash
python quant_workflow/scripts/build_qlib_direct_provider.py
```

正式五指数增量更新使用：

```bash
python quant_workflow/scripts/quant_data_cli.py update cn_index_daily --through YYYY-MM-DD
```

该命令只请求 `SH000300`、`SH000905`、`SH000906`、`SZ399006` 和 `SH000688`，按 XSHG 交易日解释 `--through`，校验七日重叠后原子发布 `cn_index_daily`；不会下载股票，也不会改旧 `.bin`。`update_qlib_index_benchmarks.py` 只保留给明确指定的旧物理 Provider 回放，不是正式更新入口。

该命令解析当前 `ashare_daily_qfq`，写入：

```text
runtime/qlib_direct/providers/ashare_daily_qfq/
  <stock_snapshot_id>/<membership_snapshot_hash>/<index_snapshot_identity>/<mode>/
    CALENDAR.parquet
    MEMBERSHIP_INTERVALS.parquet
    CLEANING_PLAN.json             # 仅 cleaned 模式
    PROVIDER_DESCRIPTOR.json
```

这里没有 `features/`，也没有 `.bin` 行情。当前 canonical 正式描述符目录实测约 92 KB，全部 Direct Provider 元数据约 660 KB；作为对照，保留的物理 `qlib/cn_data_ashare_v2` 目录约 5.2 GB。前复权价格已经在 `ashare_daily_qfq` 中完成，因此 Provider 对 Qlib 暴露的 `$factor` 在有行情的日期为 `1.0`，不会再次复权；无行情日期保持 `NaN`，不会向前填充。

描述符记录行情快照 ID、内容哈希、文件索引哈希、股票池快照哈希和适配器版本，并把股票池区间复制为一份很小的冻结证据。一个训练进程启动后只读该描述符锁定的行情与股票池，即使 `LATEST.json` 或正式股票池随后更新，也不会切到新数据。每只股票第一次读取时会核对其 Parquet 哈希。已有描述符再次启动只做快速身份核对；完整 5516 只股票验证只发生在新行情快照第一次生成描述符时。

真实验证已覆盖基础字段、滚动表达式、两进程读取和 Alpha158 的 158 个因子。对 3 只股票、8 个字段/表达式的对照结果与同源旧 `.bin` 完全一致；当前正式 direct smoke 按每只证券自身的有效行情日期抽取最多 5 个历史点，覆盖 5 个指数的 `$open/$close` 和个股的 `$close/$volume/$factor`，任一必需字段为空都判定失败。

当前迁移已经完成到正式运行入口：canonical/cleaned 训练、预测、回测、交易日历和质量报告都通过 `_paths.init_qlib_workflow()` 读取 direct Provider；五个正式指数来自 `cn_index_daily`，不是旧 `.bin`。旧 `qlib_ashare_v2`、`qlib_ashare_cleaned_v2` 只保留作兼容、审计和回滚，不能被默认解析。清洗模式不是另一份行情数据库，而是同一份 qfq Parquet 加一个哈希锁定的读取规则。后续调用方必须使用 `_paths.init_qlib_workflow()`，不能把描述符目录交给普通 `qlib.init(provider_uri=...)`；普通初始化不会自动加载自定义 Provider。

### 兼容输出退役边界

本轮不删除旧 `.bin` 数据。清单将这些输出登记为 `role: legacy` 和 `lifecycle: compatibility_output`；`status: ready` 只表示旧程序仍可读取，不代表它们是正式主入口。正式代码不再默认引用这些目录；只有 `build_qlib_v2.py`、`build_cleaned_qlib_v2.py`、`compare_native_qlib_benchmarks.py` 等明确标注的兼容/审计工具可以读取或生成它们。生产回滚只能在人工明确指定并通过完整校验后使用旧原生快照；不能把旧兼容目录重新设为 DSA 的训练、预测或回测入口。不得改写或删除原生 Parquet 快照和 `_old` 审计数据。

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
