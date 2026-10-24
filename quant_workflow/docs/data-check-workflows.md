# 本地数据检查流程

数据基础规则由轻量 `quant_qmt_data` 和已有检查脚本承担；策略对照留在 DSA。所有独立诊断只生成报告，不接封存服务，不自动下载或改行情，也不作为普通读取的全库门禁。

## 现有流程与能力边界

| 流程 | 入口 | 检查逻辑 | 能发现什么 / 不能证明什么 |
| --- | --- | --- | --- |
| 数据状态 | `quant_qmt_data.inspect_datasets`，网页 `/data` | 核对根清单已提交版本、清单摘要、数据集日期和状态 | 未登记、版本身份或元数据不一致；没有逐行证明价格正确 |
| 快照与配对校验 | `quant_data_cli.py validate <dataset_id>` | 摘要、文件索引、日期键、OHLC、raw/QFQ/因子覆盖及适用复权关系 | 文件损坏、重复、非法值、配对和算术矛盾；不能自证供应商事件真实或穷尽所有缺日 |
| ETF 只读体检 | `quant_data_cli.py check-etf` | 扫描选定 ETF 版本，重放适用日历、因子、拆分和已核读公告材料 | 按证券/日期/字段区分确定错误、证据不足、待核线索；默认不联网，不能自动认证公告真伪 |
| 版本增量对照 | `compare-snapshots`、`sample-increment` | 对照冻结前后版本、历史保留或重算说明；固定种子对照本地原始 bundle | 意外历史改写、新增记录或来源不一致；抽检不等于全检，适用已有 A 股增量契约 |
| Qlib 消费抽查 | `check_qlib_data.py` 及任务预检 | 核对所用 Provider、身份、日历和必需字段，读取固定样本 | Provider 断链、样本缺值及字段不可读；不是全市场行情真值认证 |
| 随机策略核查 | `quant_data_cli.py check-strategy` | 固定两套快照、证券、日期和参数，字段对照后用统一计价进行策略探测 | 单位/字段差异、共同异常线索、信号/持仓/诊断收益敏感性；不能把复权口径差异直接判成错误 |

`report_data_quality.py` 仍保留来源覆盖、重叠对照和 Qlib 质量报告，主要服务旧研究流程及历史报告复现。它不是新的统一门禁，也不能用它的旧“可训练”结论代替当前任务对实际本地输入的预检。`check_qlib_data.py` 当前固定抽查六个股票/指数代码的最近五日，不能把通过说成整个股票池或 ETF 都通过。

ETF 体检的既有规则与证据格式见 [ETF 文件只读体检](etf-data-healthcheck.md)。该旧入口仍按其指定版本/指针工作；新的随机策略入口从根清单冻结已提交身份，不依靠目录时间或较新的 `LATEST` 提示挑版本。

## 可调用的随机策略核查

首版复用 `quant_qmt_data.bar_snapshot/read_bars` 与 `src/core/strategy_backtest.py`，只采用轻量动量探针，不默认训练 LightGBM，不引入新回测引擎、服务或状态库。两套输入是本机可读的合法快照根；参考根可以是已合法迁入的另一台机器数据，本命令不连接 HOME、不生成输入副本。

```bash
.venv/bin/python quant_workflow/scripts/quant_data_cli.py \
  --root /path/to/primary-data check-strategy \
  --reference-root /path/to/reference-data \
  --market ashare --start 2021-01-01 --end 2026-10-23 \
  --batches 3 --sample-size 16 --sessions 63 --seed 20261024 \
  --output-dir quant_workflow/reports/data_checks/paired-stock-check
```

API 为 `check_strategy_data.audit_strategy_data(primary_root, reference_root, output_dir=..., start=..., end=..., ...)`，返回与 `summary.json` 相同的字典。`--market etf` 用于 ETF 配对快照，成交量比较单位是“份”；股票是“股”。`--codes SH600000 SZ000001 ...` 可收窄候选池，样本不足时显式减小批次/样本数，不重复抽样凑数。

随机证券、窗口及动量周期、调仓周期、持仓数先写入 `plan.json`，然后才读取样本结果。首版是均匀无放回抽样，没有把此前专项实验的风险富集采样伪装成普通随机。缺少另一侧历史、窗口存在无行情日期或零成交的证券保留在计划和报告，不补换，不推断它一定停牌；健康可配对证券继续。

默认采用快照中已绑定的交易所日历。旧快照没有充分日历证据时明确拒绝生成新计划；可以显式传 `--calendar /path/to/verified-CALENDAR.parquet`，复用已核读的 Qlib 或交易所日历，表必须有唯一、升序、无时区的 `date` 列。工具绑定字节 SHA256；来源和覆盖仍由提供者确认，不能把行情日期并集冒充交易所日历。计划固定实际抽中的日期，后续重放不重新挑日历或区间。

### 检查与比较逻辑

1. 从根清单冻结双方 raw/QFQ 身份，所选行情沿现有 reader 验证文件摘要；结束前重验所用索引、清单、行情及外部材料字节。第一次解析沿用现有 reader 的内容摘要核对，可能读取整版内容；它不是只读取几十行的性能承诺。
2. 对照 raw 的日期、OHLC、量额和可用辅助字段，再核对各侧 raw/QFQ 的量额一致性。不同来源不一致先记线索；不同 QFQ 不自动判错；缺日先记证据不足。
3. 另外检查成交额/成交量与日内价格范围的可疑关系。大宗交易、字段统计范围和单位均可能解释越界，所以该规则只提示，不据此直接修改数据。
4. 各侧 QFQ 生成同参数的动量信号；两侧统一使用 primary RAW 的开盘/收盘、同一股数规则和初始资金。记录评分、目标、持仓、交易及收益差异。不以利润高低、收益是否趋同选择“正确数据”。
5. 从双方一致的 raw 日期中再随机抽取待核样本；确认错误需要独立证据。LightGBM 特征/模型深度诊断仍属专项流程，动量一致不能排除模型特征错误。

收益只用于统一条件下的敏感性诊断：当前复用引擎按下一共同交易日开盘、100基础单位整手、零费用计算，不改变正式策略费用配置。它不模拟完整停牌/涨跌停可成交性或公司行动现金流，不作为实际可实现收益或投资绩效。当前冻结快照也不是历史时点数据重建。没有标签训练，不存在“标签也随特征口径切换”的混淆。

### 输出与固定重放

| 文件 | 内容 |
| --- | --- |
| `plan.json` | 双方冻结身份、种子、证券、日期、算法与参数 |
| `findings.csv` | 按批次、证券、日期、字段列出矛盾/差异/证据不足 |
| `agreed_spotchecks.csv` | 双方一致但仍待独立核实的样本 |
| `batch_N/primary.json`、`reference.json` | 评分、目标、持仓、交易和诊断收益 |
| `external_comparison.json` | 人工核读外部观察值与双方原始数据的对照 |
| `read-set.json`、`summary.json`、`REPORT.md` | 所用文件摘要、覆盖数量、结果及边界 |

```bash
.venv/bin/python quant_workflow/scripts/quant_data_cli.py \
  --root /path/to/primary-data check-strategy \
  --reference-root /path/to/reference-data \
  --plan quant_workflow/reports/data_checks/paired-stock-check/plan.json \
  --output-dir quant_workflow/reports/data_checks/paired-stock-replay
```

重放固定旧快照，不因后来发布而换成当前版本；被轮换移走或内容变化就明确失败。`--plan` 时日期、采样和参数来自该文件，不能另外覆盖日历。输出必须在两个数据根之外，拒绝覆盖非空报告。

外部核验可用 `--evidence observations.json`，格式如下。数值须由调用者从原始材料核读提取，工具仅复核字节绑定及比较，不能认证来源独立性、原文真伪或自动决定修复：

```json
{"observations": [{"symbol": "BJ920263", "date": "2022-04-12", "basis": "raw", "field": "volume", "value": 733481, "unit": "shares", "source_url": "https://provider.example/original", "evidence_path": "original.json", "evidence_sha256": "完整SHA256"}]}
```

价格/成交额单位为 `CNY`，股票成交量为 `shares`，ETF为 `units`，涨跌幅为 `percent`。原文路径相对证据 JSON 所在目录，拒绝越界、符号链接、错误摘要和重复观察键；范围外观察明确记为未检查。

退出码0表示诊断在请求样本范围内完成，不等于无错误；存在不可配对样本或无法完成时为2。部分可用结果仍保存。报告不修数据；确认后修复仍须显式生成新快照、保留旧版和原复权口径。删除报告不会改变行情或读取权限。
