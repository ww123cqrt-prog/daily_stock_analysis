# MiniQMT 默认行情入口

DSA 的沪深股票及 ETF 日线、实时快照优先使用已配置的 MiniQMT 行情桥。ETF 文件更新命令、网页的“更新统一 ETF 数据”任务、每日更新和兼容下载脚本共用现有增量入口，默认来源为 MiniQMT。数据仍写入外置数据目录的 Parquet 快照；回测直接读取冻结快照。

这不启动封存的 quant_data 项目服务，不读取其控制库，也不接账户或交易 API。股票分析的数据源管理器与 ETF 整池文件更新是两个消费者，共享同一行情桥；A 股全市场离线 bundle、财务、新闻、港美股等入口没有因这次接入而更换。北交所行情保留现有来源。

## 配置

Mac 无须安装 xtquant，通过已有 SSH 路线调用 HOME 的 `qmt-strategy-bridge`。HOME 需已登录行情客户端。程序不会自动重启客户端、安装 SDK 或改为交易模式。

连接地址、SSH 密钥路径等只从 `remote-windows-workstation` 的私有 `connection-targets.env` 读取；已核实的 Python、PowerShell、桥接项目根和 MiniQMT 行情端口分别来自同目录 `machine_configuration.yaml` 的 `machines.<target>.software.python.exe`、`software.powershell7.exe`、`projects.qmt_strategy_bridge.root`、`software.xtquant.market_port`。不把这些实际值复制进项目。

可选环境变量（也可放在本机 `.env`）：

| 变量 | 默认值 | 用途 |
| --- | --- | --- |
| `MINIQMT_ENABLED` | `true` | `false` 关闭 MiniQMT；股票来源按原有顺序工作，ETF 需显式指定其他 provider |
| `MINIQMT_TARGET` | `HOME` | 已有私有工作站配置的前缀 |
| `MINIQMT_TIMEOUT` | `30` | 每个桥接命令涵盖 SDK 加载、连接、取数和关闭的总秒数，最大 120 |
| `MINIQMT_CONFIG_DIR` | `~/.agents/private/remote-windows-workstation` | 既有私有配置目录 |

已有 `REALTIME_SOURCE_PRIORITY` 的显式设置优先；未设置且工作站已配置时，自动顺序以 `miniqmt` 开头。股票日线的 MiniQMT 优先级为 -1，其失败仍由原有管理器回退并披露实际来源。不存在私有配置时，不为普通安装注册失效的股票 MiniQMT provider。

## ETF 日常更新

在 DSA 根目录使用现有命令，数据根默认沿用 `QUANT_DATA_ROOT`：

```bash
.venv/bin/python quant_workflow/scripts/quant_data_cli.py update cn_etf_daily --through YYYY-MM-DD
```

现有网页任务、每日更新不会启动第二套更新流程。直接脚本仍支持准备指定证券、先不切换文件指针：

```bash
.venv/bin/python quant_workflow/scripts/download_cn_etf_data.py \
  --update --through YYYY-MM-DD --symbol SH513100 --prepare-only
```

显式选择备选来源：

```bash
.venv/bin/python quant_workflow/scripts/quant_data_cli.py \
  update cn_etf_daily --through YYYY-MM-DD --provider baostock
```

`--provider` 仅用于 ETF，也接受 `tencent`、`eastmoney`。ETF 更新不会在来源冲突后静默换源拼接。其他来源原有行为保留。

## 数据与故障处理

- 冻结已有快照和证券池，先读 HOME 缓存，仅下载窗口内缺失的交易日；按最多 20 只同窗口证券复用一次 SDK 连接。单个原始日线请求最多 400 个日历日，完整历史不因切换默认入口而重新下载。
- 已覆盖目标日的证券不请求；已准备的合格输入通过原缓存续接。周末和非交易日按日历选实际交易日。不要将盘中未结束的日线当作已完成收盘数据。
- ETF 的 SDK `front` 路径采用逐事件仿射关系：`R=1+stockBonus+stockGift+allotNum`，`C=interest-allotNum*allotPrice`，旧日价格按事件先后应用 `(price-C)/R`。此口径不能用 `dr` 代替现金、送转和配股字段。空事件返回不构成“全历史没有事件”的证明。
- 前复权采用 SDK 当前已知事件锚；查询旧日期时，后续分红也可能影响价格。响应的 `action_history_through` 记录本次核对的事件截止日，不把它冒充查询日之前的时点数据。
- 早于或等于窗口首日的股改事件不影响该窗口的价格，不因此拒绝短窗口核验；窗口内实际受未知股改影响的价格仍拒绝，事件身份和字段检查保留。
- 已观察过的旧因子事件发生变化时，该证券保留旧文件并披露需单独修复。普通尾部追加沿用现有有界 overlap 和 QFQ 衔接校验，不借机改写全部历史。
- SDK `volume` 按沪深现金证券的“手”解释；股票分析转换为“股”，ETF 旧文件保留“手”。amount 保持元，QFQ 不改变成交量和成交额。停牌标记行不被伪造成新交易 bar。
- MiniQMT ETF 更新默认逐证券隔离失败，健康证券继续；异常证券的旧文件原样保留，状态写 `published_partial` 或 `blocked`，列出错误及其实际截止日。缺口、坏值、身份或证据篡改不因此被认定合格，也不增加全库读取门禁。

桥接响应校验精确 request、request SHA、data SHA；原有缓存另绑定 base manifest、证券、日期和规则。批量响应按 SHA 保存一份，在新快照的 `source_receipts/miniqmt/` 中保留本轮来源引用。源端 SDK 内部请求数不可观察时，`network_requests` 为 null；`bridge_calls` 与 `calendar_bridge_calls` 记录桥接调用，响应的 `network_download_requested` 说明是否请求了补缺。

超时由远端命令和 SDK worker 的已有 Windows Job Object 有界收尾；不按名称结束 MiniQMT。只有文件快照更新使用原单实例锁和原子 LATEST 切换；没有项目版 Active、候选审批或发布操作。

## 已有股票比例复权的衔接

旧股票快照中的 `raw × factor`、零偏移价格属于比例复权，不能直接对这些历史价格减去新现金分红。SDK 的 `front` 与 `front_ratio` 是不同口径；即使最近价格接近，现金减法也可能把很早的低复权价变成负数。

`incremental_update_ashare_v2.merge_miniqmt_ratio_increment` 用于已冻结股票输入的显式衔接：调用方绑定原文件和来源回执，函数核对旧 raw/factor/QFQ、原始行情 overlap、明确除权事件与 SDK `front_ratio`，只追加实际取得的行情。历史价格按新事件的 `1/dr` 比例调整，保留原来已经证明的四位小数表示；不把现金式历史自动转换为比例式历史。

已证明按四位小数保存的证券，从 raw 和新因子重新按四位小数计算，追加段使用相同精度；不反复放大旧价格的舍入误差。原本全精度满足因子关系的历史保留全精度。可选的 `symbol/offset` 列原先不存在时继续缺省，不在追加段单独新增而令旧行变成空值。

发现旧窗口内遗漏的事件时，必须显式列入 `repair_event_dates`，且旧因子在事件边界没有变化、修复后完整 overlap 与 SDK 结果一致，才允许补回。不能用这项修复为未知修订兜底。新的快照保留原快照、原始响应和逐证券修复说明，不覆盖旧文件。

源端缺行单列为未解释缺日，已核实的其他行仍可供本地读取；没有证券资料不等于已经证明退市。该衔接函数不增加全库读取门禁，也不代表已接通新的全市场股票 CLI 或北交所默认桥接能力。

## 验证与回滚

定向测试覆盖 MiniQMT 适配、缺日、复权、缓存、来源替换、失败隔离、默认路由及现有任务入口。实测和离线替身分别记录；不以测试成功替代供应商数据事实。

回滚默认股票来源可设 `MINIQMT_ENABLED=false`；已有显式实时顺序仍有效。ETF 可用上述 `--provider`，历史快照不删除。代码回滚与数据回滚是不同操作；本次接入不改写已有正式快照。
