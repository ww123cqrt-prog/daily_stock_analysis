# ETF 文件只读体检

从 DSA 仓库根目录运行：

```bash
.venv/bin/python quant_workflow/scripts/quant_data_cli.py check-etf
```

显式指定目录：

```bash
.venv/bin/python quant_workflow/scripts/quant_data_cli.py \
  --root ../quant_data check-etf \
  --output-dir quant_workflow/reports/data_checks/my-etf-check
```

它读取 `cn_etf_daily/LATEST.json` 指定的一个版本，检查 raw、QFQ、因子和文件清单。默认报告写到 `quant_workflow/reports/data_checks/etf-时间/`；拒绝覆盖非空报告目录，输出必须在数据根之外。

本入口不启动项目版服务，不打开控制库，不修改行情、指针、来源或策略，也不下载历史。它不被读取、回测、训练或发布流程自动调用；报告不是这些流程的新门禁。

## 报告怎么看

| 文件 | 用途 |
| --- | --- |
| `REPORT.md` | 中文概览和检查边界 |
| `findings.csv` | 按证券、日期、组件、字段列出异常及依据 |
| `symbols.csv` | 每只 ETF 的日期、行数和各类问题数量 |
| `summary.json` | 版本、规则、计数、耗时、内存及外部证据对照 |
| `read-set.json` | 实际读取的文件、字节数和 SHA256 |

`confirmed_error` 是已经存在的矛盾，例如摘要错误、重复日期、非法价格、高低价关系错误、QFQ 改变成交量，以及与明确拆分证据冲突。

`insufficient_evidence` 是无法证明，例如成交额缺失、没有解释缺日的日历/停牌材料、因子从同组价格反推。未知值不补零；它不等于价格已经错了。

`needs_review` 是核实线索，例如 QFQ 单日变动超过 30%，或与另一来源同日同口径价格不同。原因可能是折算、分红或真实行情，不自动挑选“正确来源”。

数值错误逐日列出。同类连续记录的缺失和因子证据缺口，以起止日期及实际行数汇总；范围指文件中的记录，不表示每个自然日应有行情。计数是报告条目数，不能当成证券数或错误行总数。

`status=completed` 只表示体检结束，不表示全部正确。退出码 0 表示完整产生诊断，发现问题仍可返回 0；输入变化或无法完成时返回 2。读取与更新行为不受它影响。

## 规则来源与适配

复用 `quant_data_contract.py` 的基础行情检查、数值容差及 raw×factor+offset 关系。提取封存代码 `quant_data@a1eb38bd86c2c8830bb91a3f7cf5c79e28a2e250` 中的数据规则：

- `contract.validate_raw_qfq_bar_pair_v3`：日期、成交量及成交额一致。
- `contract.validate_canonical_bar_price_auxiliary_v1`：声明为同口径前收盘的辅助字段匹配前一条收盘价。
- `production_v2_fragments._assert_qfq_values`：复权输入、数值关系与末日基准一致。

没有引入候选、审批、receipt 闭包或全局 issue 门禁，也不在运行时依赖封存目录。加减型复权保留 offset，不强改成纯乘法；有原始份数及单位声明时检查整数手转换与半手向上取整；缺原始单位证据时不冒充已独立验证。

哈希和解码共用同次读到的字节，按证券释放 DataFrame；整体摘要复用已有摘要，不重新解码历史。结束前核对已读文件身份，标记输入是否稳定，不给后续读取授予永久资格。

## 可选证据

`--calendar calendar.csv` 或 `.parquet` 接受 `date` 和可选 `is_open`（true/false 或 1/0）。调用者须核实来源与覆盖；有该输入才对明确关闭的日期报错。没有独立日历时，只与其他 ETF 实际日期比较缺口，不当成官方停牌证明。

`--split-evidence verified-splits.csv` 重放已核实的纯拆分/合并，字段为：

```text
symbol,effective_trade_date,action_type,ratio,source_url,evidence_path,evidence_sha256
```

`action_type` 为 `split` 或 `reverse_split`；`ratio` 是每一旧份变成多少新份，1 拆 8 写 8，10 合 1 写 0.1；`effective_trade_date` 为变更后首个交易日。证据路径相对于 CSV，不能用符号链接，摘要须匹配。工具不认证公告真伪，不决定比例；输入前须核对证券、日期、比例以及是否仅含纯拆分，混合分红不能硬套。

```bash
.venv/bin/python quant_workflow/scripts/quant_data_cli.py \
  --root ../quant_data check-etf \
  --split-evidence /path/to/verified-splits.csv
```

算术自洽不能自证公司行为正确。例如 raw/QFQ 相同、因子为 1 也能通过公式；若存在真实的 1 拆 8，仍然漏了复权。独立拆分记录用于识别这种矛盾，不能以价格跳变反推比例后强行修正。

快照已声明并冻结的腾讯正文会离线重放，按证券、共同日期和四个 OHLC 字段比较。样本只支持实际窗口，不证明全历史。需要新增核验时，将有限材料放在报告目录，旧行情不动；默认一键检查不联网。

## 回滚与范围

删除报告不改变数据或使用资格。代码回滚仅撤销 `check-etf` 命令及独立实现；其他读取、下载和发布入口不依赖它。本功能检查 ETF 日线，不宣称覆盖所有股票、海外行情、财务或实时数据。
