# ETF 文件增量更新与直接读取

本入口维护旧版 `cn_etf_daily` 的 raw、QFQ、因子 Parquet 和快照索引。它只更新文件数据集的 `LATEST.json`，不启动项目版数据服务，不创建正式 candidate，也不改变 `ACTIVE_RELEASE_SET.json`。

独立检查现有文件可运行 `quant_data_cli.py check-etf`，见[ETF 只读体检](etf-data-healthcheck.md)。它只生成诊断报告，不作为更新、读取或回测的新门禁。

## 更新现有证券池

在 DSA 仓库根运行，填写数据根和已经结束的目标交易日：

```bash
.venv/bin/python quant_workflow/scripts/download_cn_etf_data.py \
  --update --quant-data-root /path/to/quant_data --through YYYY-MM-DD \
  --provider tencent --keep-failed-history
```

证券池固定为旧快照实际文件索引中的证券。更新只请求缺失尾部及默认 14 个日历日的重叠窗口，不重新下载已有全历史。重复执行已经覆盖的目标日不访问供应商。采集缓存绑定原快照摘要、证券、日期窗口和供应商，复用前重新核对文件摘要及 raw/QFQ 关系。

`--keep-failed-history` 允许健康证券继续更新；失败证券保留原文件，并在新清单和 `runtime/status/cn_etf_daily_update_latest.json` 中列出原因与原截止日。因此，`published_partial` 表示文件快照已切换、仍有证券未更新，不能解释为所有证券均已覆盖目标日。空响应、未达到目标日的响应和不一致的历史均不能产生“已完成”结论。

腾讯不可用时，可显式选择 `--provider eastmoney` 补取当前快照中仍缺日期的证券。它请求同一有界窗口的未复权价和前复权价，并保留真实供应商标识。两源数据不能仅因能够拼接就采用：旧 raw 重叠值必须一致，QFQ 必须通过下述衔接和全文件检查。

也可选择 `--provider baostock`。该入口要求现有环境已经安装真实 Baostock SDK；每次 SDK 请求在自有子进程中完成，期限覆盖登录、分页、查询和登出。SDK 的原字段和字符串响应保存到增量缓存的 `SOURCE_RESPONSE.json`，并与完整标记绑定摘要；它不是 TCP 原始报文录包。缓存复用仍校验响应及 Parquet 摘要，不因目录存在就采用。

Baostock 成交量以份计，旧 ETF 文件使用整数手。本入口按 `floor(份数 / 100 + 0.5)`（半手向上取整）统一到旧精度，同时保留 `native_volume` 和 `volume_unit`；成交额保留真实元值。旧重叠成交量仍须精确一致，不扩大比较容差。停牌占位行不追加为行情，原 SDK 响应仍保留；没有目标日实际交易行时继续列为未完成，不能伪造最新价。

## 先检查少量证券

```bash
.venv/bin/python quant_workflow/scripts/download_cn_etf_data.py \
  --update --quant-data-root /path/to/quant_data --through YYYY-MM-DD \
  --base-snapshot /path/to/quant_data/parquet/etf_1d/cn_etf_daily/snapshots/BASE \
  --symbol SZ159915 --symbol SH513100 --symbol SH512890 --prepare-only
```

该命令打印经过校验的 staging 快照路径，不切换 `LATEST.json`。未选中的证券保留原文件。只更新部分证券的准备结果不能作为整池发布。之后执行整池更新时，可复用同一冻结输入下已经取得且校验通过的采集缓存。

## 复权衔接和读取

旧 raw 历史保持原值。若供应商当前 QFQ 基准变化，只允许通过重叠区间证明统一的正比例／平移关系，再在本地转换旧 QFQ；沿用既有 OHLC 残差容差，成交量和成交额不复权。缺少重叠、原始价被修订、非统一历史变化或转换后价格非法时，拒绝该证券增量，不自动重取全历史。

衔接后重新生成并检查逐日 raw/QFQ 变换、因子、对象摘要、行数、日期和文件索引。它证明本次文件满足旧快照契约，不等于独立判定供应商所有历史价格正确。不同来源的历史版本应保留，不因回测收益高低选择数据。

研究脚本可以用 `pandas.read_parquet(snapshot / "qfq/SZ159915.parquet")` 读取指定不可变快照；需要未复权价时读 `raw/`。先固定快照清单摘要，再运行回测。历史旧快照继续可读，回滚研究输入时显式指定旧路径即可，无需覆盖新文件。

注意旧文件成交量单位应按来源核对；不能把“手”和“股”直接比较。腾讯日线接口缺少成交额时，新行的成交额保持缺失，不以价格乘成交量伪造。停牌、退市或源端缺行也不补造价格。
