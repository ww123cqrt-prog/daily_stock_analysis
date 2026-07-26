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
| A 股 Qlib v2 (`qlib_ashare_v2`) | `canonical/ready` | 由正式 qfq 快照生成，并锁定准确的上游快照 ID 和内容哈希 |
| 旧 JQData/东方财富/AkShare、merged/cleaned/Qlib | `audit/legacy_unknown` | 旧流程兼容入口，不得冒充新的正式 v2 数据集 |
| PIT CSI800 和参考表 | `canonical/ready`（完整时） | 由原有事务锁和内容哈希保护，仍在逐步迁移到统一快照结构 |

首个正式 A 股 v2 快照覆盖到 `2026-07-24`。后续是否已更新到更晚日期，必须以 `quant_data_cli.py describe` 和根 `DATA_MANIFEST.json` 为准，不能把本文日期当成动态状态。旧 A 股数据没有被移动或删除，仍作为兼容和审计数据保留。

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

# bundle 安全传回并验证后，raw/qfq 两个指针作为一组切换
python quant_workflow/scripts/quant_data_cli.py update \
  ashare_daily_qfq --bundle-dir <verified-bundle>
```

任意股票 raw、因子或 qfq 抽样核对失败，整个 bundle 都不能发布。raw 和 qfq staging 都通过哈希校验后才切换两个正式指针；任一指针或根 `DATA_MANIFEST.json` 写入失败时，两个指针、根清单和 staging 全部恢复到更新前状态。

## Qlib 血缘

新生成的 Qlib 目录必须包含 `PROVIDER_MANIFEST.json`，记录准确的行情快照、价格口径和转换配置。训练启动后锁定该 provider；数据更新不会改变正在运行的训练输入。

```bash
python quant_workflow/scripts/build_qlib_v2.py --publish
```

该命令只解析 `ashare_daily_qfq` 的 `ready/qfq` 快照，生成不可变 `qlib_ashare_v2`，并将精确的上游快照 ID 和内容哈希写进 provider 清单。Qlib 指针与根清单使用同一回滚事务。

`qlib_ashare_v2` 是当前新的正式 Qlib 入口；旧 `cn_data_baostock_jqdata*` provider 继续作为兼容数据保留。读取方应通过 `quant_data_cli.py resolve qlib_ashare_v2 --require-basis qfq` 解析正式快照，不得根据目录名自行选择 provider。

## 失败规则

- raw 下载成功、复权失败：qfq 正式指针不前进。
- 两只 ETF 只成功一只：不发布。
- 因子日期不覆盖行情、哈希变化、公司行为缺失：拒绝解析或发布。
- 请求 qfq 却解析到 raw：明确报错。
- 清单过期、README 漂移或出现未登记目录：`sync_data_manifest.py --check` 返回非零。
- 旧数据清理、移动或删除：必须另行 dry-run 并取得用户明确确认。

运行状态和日志不属于正式数据：状态写入 `quant_workflow/reports/`，日志写入 `quant_workflow/logs/`。账号、密码、token、券商信息、行情数据、模型和 `mlruns` 不进入 Git。
