# Mac QMT：SDK 主线与显式离线镜像

## 冻结输入优先

已有选定的完整 `QMT_DATASET_DIR` 时，直接离线读取 `bars.parquet` 和 `dataset.json`，不连接 SDK 或镜像。固定输入失败不会悄悄换成新数据。既有 Run 继续绑定原冻结输入，不因取数模式改变而换源。

## 当前默认：HOME 一份原生库，两端共用官方 SDK

HOME 本机继续 `QMT_TRANSPORT=local`，不需要 SSH，默认不下载，原有读取/导出入口不变。Mac 配置 `QMT_TRANSPORT=ssh`、`QMT_REMOTE_TARGET=HOME`，清除 `QMT_MIRROR_DIR`；可明确设置 `QMT_DOWNLOAD_MISSING=true`。连接和软件路径沿用工作站技能的私有配置，不在项目复制机器配置。

未选定输入时，SDK 主线按证券、日期和既有口径请求 HOME；HOME 已有数据直接读取，确需补的头尾窗口由 QMT 官方下载接口写入 HOME 原生库，再读取结果返回 Mac。Mac 仅冻结本次有限实验输入，不另建全市场库。SDK 主线由官方 SDK 计算复权；HOME 不因 Mac 的补缺开关改变本机默认行为。

Mac 当前股票与 ETF 的新请求统一采用等比前复权 `front_ratio`，本机 `.env` 的 `QMT_ADJUSTMENT`、`QMT_STOCK_ADJUSTMENT`、`QMT_ETF_ADJUSTMENT` 均设为该值；HOME 配置不变。既有固定实验按其冻结口径重放，不随默认配置重算，跨口径输入仍明确分开。

内部稀疏日不盲目下载，上市前和休市范围不制造行情。QMT 补完仍无记录时单列未知原因，网页/CLI 不将其冻结为完整输入；已有正常固定输入仍可使用。

SDK 失败直接报错，不会自动选择镜像。一次请求和实验只用一个来源，不逐行混合 SDK 与镜像数据。

## 可选备用：显式离线镜像

镜像只读取已有 `MIRROR.json` + `objects/` 快照，不需要 SDK 或 SSH。该路线保留 DAT 解析器，在本地独立计算 `none/front/back/front_ratio/back_ratio`，不依赖 SDK 内部复权实现；它与默认的 SDK 计算路线需要明确区分。

在本机 `.env` 中显式选择镜像，并关闭下载：

```dotenv
QMT_TRANSPORT=mirror
QMT_MIRROR_DIR=/path/to/existing-qmt-mirror
QMT_DOWNLOAD_MISSING=false
```

CLI 沿用现有 `read` / `export` 命令，没有新增镜像下载或同步命令：

```sh
python -m quant_qmt_data read --symbols SH600519 --start 2024-01-01 --end 2024-12-31 --adjustment front_ratio
python -m quant_qmt_data export --symbols SH600519 --start 2024-01-01 --end 2024-12-31 --adjustment front_ratio --output ./data/experiment-mirror-001
```

Python API 使用 `QMTClient(transport='mirror', mirror_dir=path)`。导出后可将新目录设为 `QMT_DATASET_DIR`；网页、Qlib 和 DSA 回测仍消费相同格式的冻结输入。

镜像读取拒绝 `download_missing=True` 和非空 `compare_mirror_dir`；CLI 对应的 `--download-missing` 和 `--compare-mirror` 同样无效。缺失证券或日期须先在 QMT 中补齐，再更新镜像。镜像不会向 SDK 借行补缺，也不从镜像下载数据。本轮不新增镜像创建、同步服务或全市场副本，不恢复旧数据平台。

镜像按各原生对象已核实的 `completed_through` 判断完整日线，不能由全局同步时间或读取时刻推定。旧清单没有逐对象完成界限时，默认排除各对象最后一行；返回的完成范围及日历随之收窄。该界限只应根据对应原文件 SHA 的核验填写。显式读取未完成行会保留标记，不能冻结为训练或回测输入；既有选定冻结输入仍直接离线复用。

## 来源身份与复权

冻结的 `dataset.json` 保留实际来源：本机 SDK 为 `source_kind='qmt_sdk_local'`，SSH 为既有 `qmt_sdk_ssh`，镜像为 `qmt_native_mirror`。镜像还保留 `mirror_dir`、`mirror_manifest_sha256`、`mirror_synced_at` 和 `source_native_sha256`，并记录所选复权口径。`mirror_synced_at` 是清单中的同步时间，不等于行情覆盖到该日。

股票/ETF 继续各自保留选定的复权口径；比较两条路线时，应显式选择同一口径或使用同一份冻结输入。本地镜像计算不代表已经证明与所有 SDK 版本和日期窗口完全一致。

## 返回 SDK 与可选对比

返回 SDK 时，清空 `QMT_MIRROR_DIR`，Mac 选择 `QMT_TRANSPORT=ssh`，HOME 选择 `local`；HOME 继续默认 `QMT_DOWNLOAD_MISSING=false`。只遗留镜像目录而未显式选择 `mirror` 仍会被拒绝。重放旧实验继续显式选择原输入路径，旧库和镜像样本不移动、不删除。

SDK 模式仍可通过 CLI `--compare-mirror` 或 API `compare_mirror_dir` 启用可选对比。差异按证券/日期/字段记录并警告，版本不同和证据不足分开标记，不改 SDK 主结果，也不阻断正常读取。

茅台现金前复权单日窗口差异保持已知暂缓，不改变正式原始行情或擅自换口径。
