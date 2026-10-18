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
| `explained_absences.csv` | 分列公告解释与供应商零成交记录；后者不等于独立认证无成交，不填造 bar、不推进实际末日 |
| `summary.json` | 版本、规则、计数、耗时、内存及外部证据对照 |
| `read-set.json` | 实际读取的文件、字节数和 SHA256 |

`confirmed_error` 是已经存在的矛盾，例如摘要错误、重复日期、非法价格、高低价关系错误、QFQ 改变成交量，以及与明确拆分证据冲突。

`insufficient_evidence` 是无法证明，例如成交额缺失、没有解释缺日的日历/停牌材料、因子从同组价格反推。未知值不补零；它不等于价格已经错了。

因子标记为 `derived_close_ratio` 时也只是价格反推，不能当成独立公司行为证据。逐行 `source_id` 可以追溯到已冻结的旧快照文件；这只证明该行从哪里继承，不会把旧快照自动升级成供应商原始回执或停牌证明。

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

A 股旧 TickFlow 因子表的空 `symbol`/`offset`，可通过 `incremental_update_ashare_v2.restore_tickflow_factor_metadata` 做显式离线修复。调用方须先核对冻结清单、索引和文件摘要；函数只补已声明为 `tickflow_adj_factor` 的缺失单元，证券须与已知字段一致，填入偏移量后四个 OHLC 必须与保存的 raw/QFQ 逐键吻合。无法证明就拒绝修复，不改已有价格、非空字段或省略了可选列的合法旧表。修复结果写入新快照，旧文件保留；正常读取和体检不会自动调用它。增量工具在追加单位因子时沿用已声明的证券和零偏移列，避免新行再次因拼接丢失这些字段。

拟合因子存在舍入差异，平价行情的仿射表示也可能不唯一。这类比例差异只列待核查，不直接判错；明确拆分且边界两侧 QFQ 仍完全等于 raw 时，才归为确定漏复权。

快照已声明并冻结的腾讯正文会离线重放，按证券、共同日期和四个 OHLC 字段比较。样本只支持实际窗口，不证明全历史。需要新增核验时，将有限材料放在报告目录，旧行情不动；默认一键检查不联网。

已核读的停牌、清算或终止上市公告可随新派生快照保存。清单的 `source.evidence_followup` 引用快照内 JSON 的 `path`、`sha256`；该文件声明 `dataset_id=cn_etf_daily`、`checked_through` 和 `absence_intervals`。每段包含 `symbol/start/end/reason`、当前 `raw/qfq` 的 `component_sha256`、`sources`（公告 `path/sha256/url`）及核读 `note`。原因仅接受 `trading_suspended`、`fund_liquidation`、`delisted`。

这是人工核读结果的有界留存，和拆分 CSV 一样，工具只复核文件、摘要、证券及日期绑定，不自动认证公告真伪或推断日期。材料缺失、被替换、价格文件已变或范围不符时，缺日仍列证据不足。已解释日期另列，实际文件末日和目标达标数不改。MiniQMT 空缓存、空因子表、周末的 `IsTrading=False` 或零成交观测，均不能代替停牌公告。补到部分因子事件也不证明其余历史不存在事件；第三方对照按实际证券、日期、字段留存。

若因子已从供应商实际公布的表取得，可用 `source.factor_registry_evidence` 按证券保存新浪表的 `path/sha256/as_of/component_sha256`。绑定包括当前 raw、QFQ、因子三个文件。体检离线解析对应证券的 `qfq.js`，检查完整条数、日期唯一、支持的变换类型和有限正比例，再按实际文件末日归一；逐日核对四个 OHLC 及因子、偏移。只有这些都吻合，且因子明确标为 `sina_qfq_registry`，才移除该文件“仅从同组价格反推”的提示。只改标签、空回复、缺文件、摘要/范围不符或数值不同均不能获得该结果。

这证明所存供应商因子表支持这些数值，不认证供应商穷尽了所有真实事件；MiniQMT 空列表仍不单独证明从未分红。`front` 的加减式前复权和 `front_ratio` 等比前复权有不同数值，后者不能为消除前者的负数而静默替换。缺日补充必须是实际保存的来源记录，成交量为零不等于程序可以凭前收盘填造 bar；来源明确报停牌或复权关系未核实时分别披露，不阻断其他文件读取。

缺日核实可保留 Baostock 的逐日原始响应，通过 `source.zero_activity_evidence` 绑定当前 raw/QFQ 文件摘要。体检只把实际返回的、同证券同日、原始口径、零成交量与成交额且 OHLC 相同的记录列为 `provider_reported_zero_activity`；供应商的 `tradestatus=0/1` 原样披露。这不是交易所停牌公告，也不是独立证明当日无成交，不插入假 bar、不推进实际行情末日。空响应、非零成交、身份变化或被篡改的记录不能解除缺日提示。完整说明在 `explained_absences.csv`，与正式公告证据分开。

## 回滚与范围

删除报告不改变数据或使用资格。代码回滚仅撤销 `check-etf` 命令及独立实现；其他读取、下载和发布入口不依赖它。本功能检查 ETF 日线，不宣称覆盖所有股票、海外行情、财务或实时数据。
