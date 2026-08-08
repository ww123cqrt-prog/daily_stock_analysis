---
name: quant-data-sample-validation
description: Perform a fast, read-only, deterministic spot check of governed quant_data snapshots. Use this project skill whenever the user asks for a quick data check, spot check, sampling validation, sanity check, or an immediate confidence check before research. Always report the unchecked remainder and never call a sample pass a full validation.
---

# quant_data 数据抽检

## 目的

抽检用于快速发现指针错位、文件缺失、样本哈希变化、行数/日期漂移和因子文件缺失。它不检查每个文件，也不验证OHLC数值关系或复权公式，不能证明整个数据集完全正确。

## 默认范围

- 用户指定数据集时只抽检指定目标。
- 用户未指定时，抽检所有 `ready` 且具有原生 `LATEST.json` 的正式快照。
- 默认每个数据集检查10个索引文件；证券少于10只时全部检查。
- 抽样由 `dataset_id + snapshot_id + seed` 确定，相同输入得到相同样本，方便复核。

## 命令

```bash
cd /Users/cq/WorkingProjects/daily_stock_analysis
./.venv/bin/python \
  .claude/skills/quant-data-sample-validation/scripts/sample_validate.py \
  --root /Users/cq/WorkingProjects/quant_data \
  --dataset cn_etf_daily \
  --sample-size 10 \
  --seed daily-check
```

可以重复传入 `--dataset`。不传时脚本抽检所有可解析的 `ready` 快照。

## 检查内容

1. 根清单、`LATEST.json`、快照清单的 dataset/snapshot/hash 是否一致。
2. 样本路径是否逃逸、缺失或变成符号链接。
3. 样本文件SHA256和大小是否与索引一致。
4. Parquet样本的行数、首尾日期、日期重复和日期顺序；若文件含任一OHLC列，
   只检查四列是否齐全，不检查价格大小关系或数值正确性。
5. 被抽中证券对应的复权因子文件是否存在，及其哈希、行数和自身首尾日期是否
   符合因子索引；不在抽检中证明因子与行情逐日覆盖或复权公式成立。
6. 公司行为文件存在时检查其哈希。

脚本只读，不更新清单、不修复数据、不移动文件。

## 结论口径

抽检通过时只能说：

> 本次抽检通过；检查了 X/Y 个行情或Qlib索引文件，仍有 Y-X 个文件未逐一检查。要确认整套数据，必须运行 `quant-data-full-validation`。

任一错误都要报告数据集、相对路径和具体不一致，停止训练或回测读取该快照。
