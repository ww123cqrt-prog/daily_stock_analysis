---
name: quant-data-etf-update
description: Update an existing governed ETF dataset by reusing its frozen universe and history, requesting only a bounded overlap plus missing dates, and limiting any history repair to individually proven affected symbols. Never escalate an update into a full rebuild.
---

# ETF 日期更新

## 适用范围

使用本技能把已有正式 ETF 数据更新到一个更晚的交易日。默认规则是缺什么补什么：复用全部已有历史、冻结现有证券池，只取得有限重叠窗口和缺失日期。

## 硬性边界

- 目标日期不晚于当前快照且文件完整：零网络请求，直接返回当前快照。
- 普通更新不得重新选择ETF池，不得重下所有ETF历史。
- 每只ETF只请求有限重叠窗口和缺失日期。
- 重叠核验确认某只ETF复权历史变化时，只处理该只ETF；不得连带重取其他ETF。
- 无法确认复权口径时停止发布并报告，不能自动转入全量重建。

全量重建只有在当前快照严重错误、唯一备份无法恢复且用户明确批准本次重建时才允许。任何条件不明确都必须停止并问用户。

正式证券池从当前 `ETF_UNIVERSE.parquet` 冻结继承。扩大或缩减证券池必须作为单独变更说明，不能混入日期更新。

## 开始前

读取：

1. `/Users/cq/WorkingProjects/quant_data/README.md`
2. `/Users/cq/WorkingProjects/quant_data/AGENTS.md`
3. `/Users/cq/WorkingProjects/quant_data/DATA_MANIFEST.json`
4. `/Users/cq/WorkingProjects/daily_stock_analysis/quant_workflow/docs/quant-data-contract.md`

先执行：

```bash
cd /Users/cq/WorkingProjects/daily_stock_analysis
./.venv/bin/python quant_workflow/scripts/quant_data_cli.py \
  --root /Users/cq/WorkingProjects/quant_data describe cn_etf_daily
```

向用户说明旧最新日期、目标日期、预计新增交易日和证券数。若目标日期不晚于正式快照，普通更新不联网。任何全量重建都必须另外满足“当前严重损坏、备份无法恢复、用户明确批准”三个条件。

## 更新命令

```bash
cd /Users/cq/WorkingProjects/daily_stock_analysis
./.venv/bin/python quant_workflow/scripts/quant_data_cli.py \
  --root /Users/cq/WorkingProjects/quant_data \
  update cn_etf_daily --through YYYY-MM-DD
```

159915和513100已经属于统一目标，不得再建立或更新独立两ETF数据集。任意一只ETF失败时都不得发布半套数据。

## 更新后核对

1. 比较新旧 `ETF_UNIVERSE.parquet`，普通更新必须没有新增或移除代码。
2. 校验同一快照内 raw/qfq证券数、日期和行数一致。
3. 校验分段 `factor + offset` 能复原OHLC，最新日归一为 `1/0`。
4. 确认磁盘只保留当前快照和一份上次成功快照。
5. 运行 `quant-data-sample-validation` 做即时抽检。
6. 定时任务或正式交付前运行 `quant-data-full-validation`。

## 回测与公司行为口径

ETF 研究和回测必须在报告中明确区分 `raw`、`qfq` 和真实成交价：

- `qfq` 可用于连续收益、动量和均线信号，但 `qfq open` 不能直接描述真实可成交的开盘现金价格；
- 如果回测使用 `raw` 价格，必须读取同一快照的 `factors/` 和 `corporate_actions.parquet`，并处理拆分、分红、配股、份额变化和现金流；
- 如果回测使用 `qfq` 价格，必须记录复权公式、因子/offset 证据、是否包含现金分配，以及该结果是否只是连续价格研究口径；
- 不得把 `qfq` 信号与 `raw` 成交、或 `qfq` 估值与 `raw` 持仓账务混用；跨平台验证必须统一价格基础、公司行为、成交时点和现金结算顺序；
- 公司行为处理不能只看价格跳变。必须检查持股数量、持仓成本、加仓/换仓阈值、组合市值和交易收益是否同步；
- 未验证现金分红再投资和真实成交口径时，报告不得称为“实盘可复现收益”，只能称为“复权连续价格回测”。

对于策略复核，至少保留两套对照：

1. `qfq` 连续价格版本：用于检查信号逻辑和历史收益路径；
2. `raw + 公司行为账本 + 真实成交价` 版本：用于检查可交易性和跨平台一致性。

若两套结果差异显著，必须先解释公司行为和成交口径，不得直接归因于策略 Alpha。

供应商请求、数据校验、暂存或发布事务在切换指针前失败时，明确说“没有发布，
正式数据仍指向旧快照”；不能只说缓存里下载了多少只。若指针已经成功切换，之后
只有旧快照保留清理或根清单再次同步失败，则必须报告
`published_with_warnings`：新快照已经发布，但维护告警仍需处理，不能误报为
“没有发布”。

## 完成报告

报告旧日期、新日期、新增交易日、证券数、实际请求数、复用证券数、单独重算证券数、快照ID、抽检/全检状态和总耗时。若请求数明显超过“证券数 × raw/qfq各一次”的有限窗口规模，必须视为异常并停止检查。
