# Mac 使用 HOME QMT 原生镜像

HOME 负责在 QMT 中下载行情。`quant_qmt_data sync` 单向复制已存在的日线 DAT 和必要 SDK 元数据；Mac 从已提交镜像读取。DSA 沿用 Windows 分支的字段、成交量单位、实验冻结、网页状态、Qlib 和回测入口，无需安装 Windows SDK。

## 三类位置各做一件事

| 位置 | 用途 | 读取选择 |
| --- | --- | --- |
| HOME 当前 QMT 原生目录 | 行情真源，由 QMT 维护 | 同步时由正在运行的 QMT 返回，忽略旧 MiniQMT 缓存配置 |
| `quant_qmt_data/runtime/qmt-mirror` | 原生文件的离线镜像，`MIRROR.json` 为提交点 | `QMT_MIRROR_DIR`，只读请求所需对象 |
| 本次实验目录 | 有限标的、日期、日历和身份，可复现训练/回测 | `QMT_DATASET_DIR`；损坏时明确失败，不回退其他目录 |

旧 `quant_data` 和 `quant_qmt_data/data` 本轮保持原位，不是新入口的候选来源。旧代码只在 Git 历史，不复制源码目录，也不自动加载历史分支。新入口缺失时显示缺失，不能偷偷借旧库补齐。

## 在现有环境运行

先在 `quant_qmt_data` 原目录执行同步（证券清单可以按研究需求增加）：

```sh
python -m quant_qmt_data sync --symbols SH600519 SZ000001 SH600000 SZ002415 SH510300 SZ159915 SH000001 SH000300 SZ399006 --mirror-dir ./runtime/qmt-mirror
```

DSA 进程配置 `QMT_MIRROR_DIR` 的绝对路径。Mac 股票保持 `QMT_STOCK_ADJUSTMENT=front_ratio`，ETF 保持 `QMT_ETF_ADJUSTMENT=front`；HOME 原有配置不变。若两类复权不同，应分别冻结输入，不在一份实验里悄悄混用。未设置分类配置时保持 Windows 原来的 `front` 默认行为。

已有 `load_market_input` 在没有选定输入时，从镜像读取所需证券和日期，并用 `save_dataset` 冻结到 reports 下；它只保存本次研究范围。选择已生成目录到 `QMT_DATASET_DIR` 后，`/data` 展示该输入，网页回测和训练均使用其 `experiment_id/bars_sha256/inputs_sha256/price_basis`。更换研究范围需明确选择或重新冻结，不能把一个小样本目录说成全市场数据状态。

训练沿用 `baseline_qmt_native.yaml`、`--provider qmt_native --provider-uri <input>`，不依赖旧 HTTP 服务。正式研究仍需合理选择股票池、训练/验证/测试日期；当前股票池没有历史成员时不能声称消除存续偏差。

## 本轮验收范围

在原目录、原 DSA 环境完成九个股票/ETF/指数文件的同步，重复同步零文件传输。15 组原始/等比/现金读取共 18,178 行与 HOME 官方 SDK 对齐；这是有限样本，不是全市场验收。四股票等比输入和两 ETF 现金输入分别冻结；网页、2024 小回测及 2021–2024 小训练的日志和输入身份保存在本机 `quant_workflow/reports/experiments/mac-native-mirror-20261029`，不提交行情和模型。

茅台 2006-05-25 现金前复权的 SDK 窗口差异继续作为已知问题暂缓，本轮没有修改原始行情或植入特殊修正。旧库封存留待镜像使用范围确认后另行处理。
