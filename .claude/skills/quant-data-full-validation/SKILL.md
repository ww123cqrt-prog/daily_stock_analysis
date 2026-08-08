---
name: quant-data-full-validation
description: Perform a read-only full validation of governed quant_data datasets, including every indexed file hash, schema and date metadata, adjustment evidence, latest pointers, root manifest consistency, and relevant data-contract tests. Use this project skill whenever the user asks for a full check, complete validation, release validation, post-update acceptance, corruption investigation, or proof that a formal snapshot is ready.
---

# quant_data 数据全量检查

## 目的

全检逐一读取并校验目标正式快照的全部索引文件。它比抽检慢，但正式发布、训练切换、数据损坏调查和定时任务验收必须使用本技能。

全检是只读操作，不自动修复、删除或切换快照。

## 开始前

读取 `quant_data/README.md`、`AGENTS.md`、根 `DATA_MANIFEST.json` 和项目的 `quant-data-contract.md`。记录检查开始时间、目标数据集、快照ID和文件数。

用户指定数据集时只全检该目标。用户说“整个 quant_data”时：

- 全检所有 `ready` 且有 `LATEST.json` 的原生快照；
- 对 `pit_csi800` 和 `robust_reference` 运行各自的完整性/preflight检查；
- `legacy_unknown` 只做登记和路径盘点，不能宣称已验证为正式可用数据；
- `stale` 兼容输出必须继续报告为过期，不能因文件存在而算通过。

## 单数据集全检

```bash
cd /Users/cq/WorkingProjects/daily_stock_analysis
./.venv/bin/python quant_workflow/scripts/quant_data_cli.py \
  --root /Users/cq/WorkingProjects/quant_data \
  validate <dataset_id>
```

这个命令会逐文件读取行情或Qlib索引，核对全部SHA256、行数、日期、复权因子覆盖、公司行为文件和快照内容哈希。

### ETF正式发布验收（默认快捷入口）

当目标是正式 ETF 数据集 `cn_etf_daily` 时，使用下面这一条命令完成 ETF 范围内的发布验收：

```bash
./.venv/bin/python quant_workflow/scripts/quant_data_cli.py \
  --root /Users/cq/WorkingProjects/quant_data \
  validate cn_etf_daily --release
```

该命令是只读、无网络的单次检查，会一次完成：

- 当前 `LATEST.json` 指向的完整快照和唯一上一版备份的逐文件校验；
- raw、qfq、factors、公司行为、证券池和根清单的一致性校验；
- 当前与备份各自的证券数量记录（当前正式数据实测为 51/50），以及五个注册策略标的
  (`SH512890`、`SH513010`、`SH510720`、`SZ159915`、`SH513100`) 是否在当前同一 qfq 组件；
- 旧 ETF 数据集身份、旧散落目录和多余快照检查。

它只访问 `parquet/etf_1d/cn_etf_daily` 及 ETF 相关根清单字段，不调用网络，也不运行整个 `sync_data_manifest.py --check`。输出为 JSON，成功状态为 `full_validation_passed`，失败状态为 `full_validation_failed`，并带耗时和明确失败原因。ETF 发布验收不要再手工组合本节其他命令。

## 根清单检查

```bash
./.venv/bin/python quant_workflow/scripts/sync_data_manifest.py \
  --quant-data-root /Users/cq/WorkingProjects/quant_data --check
```

ETF只全检 `cn_etf_daily` 这一个数据集，但校验器必须逐一检查同一快照内的raw、qfq、因子和公司行为，不能只检查qfq。

## 专项回归

```bash
./.venv/bin/pytest -q \
  quant_workflow/tests/test_quant_data_contract.py \
  quant_workflow/tests/test_quant_data_cli.py \
  quant_workflow/tests/test_sync_data_manifest.py \
  quant_workflow/tests/test_download_cn_etf_data.py \
  quant_workflow/tests/test_download_etf_momentum_data.py
```

测试通过证明校验程序的规则仍正常；它不能替代对当前快照文件的逐一校验。两者都要有结果。

## 完成报告

逐个数据集报告：快照ID、价格口径、证券/文件/行数、日期范围、内容哈希、全部文件检查数量、错误数量、根清单结果和耗时。

结论只能使用：

- `full_validation_passed`：目标范围全部完成且零错误；
- `full_validation_failed`：发现明确错误；
- `full_validation_incomplete`：有数据集或专项检查未执行。

不能把抽检、部分数据集通过或测试通过单独写成 `full_validation_passed`。
