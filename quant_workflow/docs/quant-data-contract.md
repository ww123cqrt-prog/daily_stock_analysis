# quant_data 数据契约

本文档是 `quant_data/` 的受 Git 管理版本。数据根目录中的 `README.md` 和
`DATA_MANIFEST.json` 用于本机接手；本文档用于代码审查、跨线程交接和仓库升级。

## 接手顺序

1. 读取 `${QUANT_DATA_ROOT}/README.md`。
2. 读取 `${QUANT_DATA_ROOT}/DATA_MANIFEST.json`，只把 `official_entries` 当作正式入口。
3. 读取目标子目录的 `README.md`。
4. 运行 `quant_workflow/scripts/preflight_robust_data.py`；报告为
   `blocked_data` 时不得启动鲁棒策略训练。

README 说明存储规则，manifest 记录当前磁盘实况，preflight 证明数据能否用于当前
鲁棒策略。三者冲突时停止任务并报告差异，不自行选择看起来可用的数据。

## 数据分层

| 路径 | 内容 | 是否正式入口 |
|---|---|---|
| `parquet/ashare_1d/` | 按股票拆分的 A 股日线 Parquet | 只有 manifest 登记的 cleaned 目录是正式入口 |
| `qlib/` | Qlib 二进制 provider | 只有 manifest 登记的 cleaned provider 是正式入口 |
| `reference/` | PIT 股票池、证券主表、行业、ST、停牌 | 只有完整事务快照是正式入口 |
| `cache/` | 可恢复下载缓存 | 否，禁止直接训练 |
| `raw/` | 供应商原始数据 | 否，必须经过规范化和校验 |
| `qlib_csv/` | 转换中间态或历史产物 | 否 |
| `parquet/etf_1d/` | ETF 衍生数据 | 未登记来源和生产脚本前不是鲁棒策略输入 |

运行状态、日志和阶段报告不属于数据湖正式数据。状态写入
`quant_workflow/reports/`，日志写入 `quant_workflow/logs/`。

## A 股日线契约

- 文件组织：每只股票一个 `<6位代码>.parquet`。
- 主键：单文件内 `date` 唯一。
- 必需列：`date, open, close, high, low, volume`。
- 可选列：`amount, turn, pctChg`；合并产物还可包含 `factor`。
- 价格口径：源目录均为前复权/预复权价格空间。不同供应商的复权曲线不能在同一
  日期逐列混合。
- 合并顺序：Baostock、JQData、EastMoney、AkShare。前一个来源已有的日期保持不变，
  后一个来源只按整行补缺失日期。
- 正式 cleaned 数据由 `apply_data_cleaning.py` 从合并数据生成，不覆盖原合并数据。

生产脚本：

| 数据 | 生产脚本 |
|---|---|
| `baostock_qfq/` | `download_baostock.py` |
| `jqdata/` | `download_jqdata.py` |
| `eastmoney_qfq/` | `download_eastmoney.py` |
| `akshare_qfq/` | `download_akshare.py` |
| `merged_baostock_jqdata/` | `merge_to_qlib.py` |
| `merged_baostock_jqdata_cleaned/` | `apply_data_cleaning.py` |
| `qlib/cn_data_baostock_jqdata_cleaned/` | `apply_data_cleaning.py` |

## 鲁棒策略数据门槛

鲁棒策略必须同时具备：

- PIT CSI800 日级成员和成员区间；
- `security_master.parquet`；
- `industry_intervals.parquet`；
- `st_status.parquet`；
- `suspension_status.parquet`；
- `REFERENCE_SOURCE.json`；
- 清洗后价格和 `SH000300/SH000905/SH000906` 基准；
- 通过 `preflight_robust_data.py` 的覆盖率、主键、前视和来源校验。

不能用今天的成分股回填历史，不能直接从 `cache/` 训练，不能因为数据不完整降低
验收门槛。

## 下载与复用

正式鲁棒策略配置默认使用：

- `universe.reference_mode: reuse_existing`：只验证并复用已经发布的参考快照，不再次访问 Baostock/JQData；
- `universe.benchmark_mode: reuse_existing`：只验证三个基准的 Qlib bin 是否覆盖完整日历，不再次访问网络。

参考数据的首次全量下载必须在隔离 staging 中完成，再校验来源、日期范围、成员股票池哈希、字段、主键和内容哈希，最后事务发布到 `reference/`。指数基准也必须先完成下载和校验，再写入 Qlib provider。校验失败时状态为 `blocked_data`，不能自动退回联网下载，也不能用部分缓存训练。

远端工作站生成的数据传回本机后仍然只是 staging，不能用 `cp`/`rsync` 直接覆盖正式目录。标准导入入口是：

```bash
python quant_workflow/scripts/download_robust_reference.py \
  --pit-dir "${QUANT_DATA_ROOT}/reference/pit_csi800" \
  --reference-dir "${QUANT_DATA_ROOT}/reference" \
  --start-date 2014-01-02 --end-date 2026-07-16 \
  --import-dir <本机参考数据暂存目录>

python quant_workflow/scripts/add_index_benchmark.py \
  --provider-dir "${QUANT_DATA_ROOT}/qlib/cn_data_baostock_jqdata_cleaned" \
  --required-start-date 2014-01-02 \
  --import-provider-dir <本机基准指数暂存 provider>
```

第一条命令重新核对本机 PIT 股票池哈希、参考表结构和时点内容，再复用参考数据下载器已有的可恢复事务发布。第二条命令要求暂存 provider 与正式 provider 的日历文件字节完全一致，验证三个指数的六个 Qlib bin 及股票池隔离后，再用可恢复目录事务安装。两条导入路径都不访问 Baostock/JQData；失败时不得改走联网下载。

后续日常更新只补新增交易日；如果 PIT 股票池历史分区发生变化，成员哈希会变化，相关参考数据必须重新生成，旧快照不能继续复用。

## 变更规则

- 新增正式数据时，同步更新数据根 README、子目录 README 和 `DATA_MANIFEST.json`
  生成逻辑。
- 多文件正式数据必须事务发布；存在 transaction 标记时读取方应拒绝使用。
- 移动、归档或删除前必须先生成 dry-run 盘点，并取得用户明确确认。
- 不把账号、密码、token、券商信息、行情数据、模型或 `mlruns` 提交到 Git。
