---
name: quant-data-etf-full-download
description: Fully rebuild governed ETF raw/qfq history only when the current snapshot is proven severely unusable, its validated backup cannot restore service, and the user explicitly approves the stated full rebuild. Never infer approval from an update or merge request.
---

# ETF 全量下载与重建

## 三重批准门槛

ETF全量重建只有以下三个条件同时成立才允许执行：

1. 当前 `cn_etf_daily` 快照经过全检确认存在严重错误，无法继续使用；
2. 唯一备份快照不存在、损坏，或实际恢复验证失败；
3. 已向用户列明证券数、历史范围、预计请求数、耗时和替代方案，并取得用户对**本次ETF全量重建**的明确批准。

“更新到某日”“补齐数据”“合并数据集”“继续”均不构成批准。只要对用户意思、数据是否严重损坏或备份能否恢复有疑问，必须停止并询问，不能自行决定重下。

缺少日期只走ETF日期增量更新；缺少某只证券只补该证券；单只ETF复权历史变化只处理该只ETF。当前快照可用或备份可恢复时，禁止本技能。

以下两件事必须分清：

- “当前正式证券池的全历史”默认指 `cn_etf_daily` 已登记的证券池；
- “交易所全部 ETF”是扩大证券池，数量和请求量完全不同，不能根据“全量”二字擅自扩大。

普通更新到昨天或指定新日期使用 `quant-data-etf-update`。第一次创建大型ETF数据集也必须先说明范围并取得用户明确批准。

## 固定事实

- ETF 只有一个正式数据集 `cn_etf_daily`；每个快照内部同时保存 `raw/`、`qfq/` 和 `factors/`。
- 第一阶段统一证券池为51只，必须包含 `512890`、`513010`、`510720`、`159915` 和 `513100`。
- 只保留当前快照和一份上次成功快照；不得重新创建两ETF专用正式入口。
- `513010` 是恒生科技ETF易方达；红利国企ETF国泰是 `510720`。
- 腾讯前复权证据使用分段仿射关系：`qfq = raw * factor + offset`。

## 开始前

依次读取：

1. `/Users/cq/WorkingProjects/quant_data/README.md`
2. `/Users/cq/WorkingProjects/quant_data/AGENTS.md`
3. `/Users/cq/WorkingProjects/quant_data/DATA_MANIFEST.json`
4. `/Users/cq/WorkingProjects/daily_stock_analysis/quant_workflow/docs/quant-data-contract.md`

先运行 `describe` 和全检，记录当前严重错误证据；再实际验证唯一备份无法恢复。缺少这两份证据时停止。向用户明确说明本次是完整历史下载、证券范围、目标日期和预计请求量，并保存用户批准记录。用户只要求50只试运行时，禁止扩大到全部上市ETF。

## 执行流程

1. 检查 `quant_data/runtime/locks/cn_etf_update.lock` 和对应缓存进度，禁止并行运行。
2. 从新浪取得候选清单，以同花顺交叉核验；历史 raw/qfq 只从腾讯成对取得，不能把不同供应商的前复权序列拼接。
3. 对代码和下载器做静态安全检查，不执行第三方社区脚本。
4. 使用全局限速、请求预算、最多两次重试和连续失败熔断。缓存按证券保存，失败后可以恢复。
5. 每只证券完整验证日期、OHLC、raw/qfq关系、分段 factor/offset 和最新日 `factor=1, offset=0`。
6. 只有全部证券成功时，才把统一 raw/qfq/factors 快照原子发布；发布事务失败时
   旧 `LATEST.json` 保持不变。指针切换后的保留清理或根清单再次同步失败只记为
   `published_with_warnings`，不得把已经成功的发布说成失败。

51只正式证券池的典型命令：

```bash
cd /Users/cq/WorkingProjects/daily_stock_analysis
./.venv/bin/python quant_workflow/scripts/download_cn_etf_data.py \
  --quant-data-root /Users/cq/WorkingProjects/quant_data \
  --full-rebuild \
  --confirm-severe-current-corruption \
  --confirm-backup-unrecoverable \
  --confirm-user-approved \
  --through YYYY-MM-DD \
  --limit 51 \
  --required-code 512890 \
  --required-code 513010 \
  --required-code 510720 \
  --required-code 159915 \
  --required-code 513100
```

默认保留断点缓存。只有用户明确要求忽略已有缓存重新下载时才使用 `--no-resume`；不要为“看起来更干净”删除缓存或历史快照。

## 发布后检查

```bash
./.venv/bin/python quant_workflow/scripts/quant_data_cli.py \
  --root /Users/cq/WorkingProjects/quant_data validate cn_etf_daily
./.venv/bin/python quant_workflow/scripts/sync_data_manifest.py \
  --quant-data-root /Users/cq/WorkingProjects/quant_data --check
```

再使用 `quant-data-full-validation` 做正式全检。不能把下载完成、缓存完成或抽检通过说成正式发布成功。

## 完成报告

必须明确报告：证券范围、成功/失败数、raw/qfq/因子行数、起止日期、请求数、复用缓存数、复权方法、统一快照ID、上一成功快照ID、验证结果、发布后告警，以及旧快照是否仍可回滚。
