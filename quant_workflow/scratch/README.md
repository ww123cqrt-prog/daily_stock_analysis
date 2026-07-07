# Scratch

Temporary or historical scripts live here. They are kept for traceability, not
as the main workflow.

- `akshare_old/` - previous AKShare/EastMoney full-download attempts and the
  old AKShare+JQData merge script.
- `experiments/` - early Qlib LightGBM experiments used while validating the
  data pipeline.
- `experiments/model_research/` - Hermes-era model comparisons such as PCA,
  LSTM, fundamental factors, IC-weighted portfolios, and realistic backtest
  probes. Treat these as research notes, not durable workflow entry points.
- `experiments/fundamental_data/` - exploratory Baostock valuation/fundamental
  downloads. These write partial CSI300-only factors and are not yet part of
  the canonical provider build.
- `experiments/baostock_extend/` - exploratory long-history Baostock extension
  scripts that overwrite Parquet files. Use the durable downloader/supervisor
  first unless intentionally rebuilding long history.
