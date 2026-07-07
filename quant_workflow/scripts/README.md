# Scripts

All stable scripts in this directory are intended to run with the shared DSA
Python runtime.  DSA passes that runtime through the quant job service; from a
shell, prefer:

`${WORKING_PROJECTS_ROOT:-/path/to/WorkingProjects}/daily_stock_analysis/.venv/bin/python`

Do not add script-local virtual-environment fallbacks unless the DSA backend
explicitly passes them as an override for a one-off repair.

Path convention:

- Runtime defaults are centralized in `_paths.py`.
- `WORKING_PROJECTS_ROOT` moves the default project family roots together.
- `QUANT_WORKFLOW_ROOT`, `QUANT_WORKFLOW_REPORTS_ROOT`,
  `QUANT_WORKFLOW_MLRUNS_DIR`, `QUANT_DATA_ROOT`, `QLIB_PROVIDER_URI`,
  `QLIB_CLEANED_PROVIDER_URI`, and `CHINA_ASTOCK_QUANT_ROOT` override specific
  paths when needed.
- Do not add new hardcoded local absolute paths in scripts.

Naming convention:

- `download_<source>.py` fetches source data into Parquet.
  `download_jqdata.py --dry-run` is the safe authentication and stock-pool
  parsing check used by the web workflow; omit `--dry-run` only when you intend
  to write real Parquet files.
- `supervise_<source>.py` runs a restartable long download.
- `merge_to_qlib.py` creates the merged Parquet and Qlib bin provider.
- `apply_data_cleaning.py` applies anomaly reports into a separate cleaned
  Parquet + Qlib provider without overwriting the canonical provider.
- `check_qlib_data.py` smoke-checks the converted provider.
- `repair_qlib_provider.py` fixes provider instrument metadata without a full rebuild.
- `add_index_benchmark.py` and `generate_csi300.py` are provider-maintenance
  scripts. They now support `--dry-run` and parameterized paths, but should
  stay at the script layer until they write report artifacts suitable for the
  DSA web UI.
- `supervise_baostock.py` is a restartable long-download supervisor. It is not
  exposed as a separate DSA web button because the web task center already wraps
  Baostock downloads with durable job logs and cancellation.
- `report_data_quality.py` generates coverage, overlap, merged-data, Qlib
  provider, and training-readiness reports. Use `--overlap-sample` to limit
  Baostock/JQData overlap comparisons and `--merged-sample` to limit merged
  Parquet scans for browser smoke checks; omit both for a full report.
- `report_workbench_capabilities.py` generates a read-only inventory of DSA
  pages, DSA API routers, web-exposed quant jobs, stable `quant_workflow`
  scripts, and `china-astock-quant` reuse boundaries. It writes
  `workbench_capability_inventory_latest.{json,md}` and never starts
  download, conversion, cleaning, training, broker, or trading tasks.
- `train_baseline_lgb.py` runs the canonical Qlib Alpha158 + LightGBM baseline.
  Use `--provider cleaned` to train against the cleaned provider while keeping
  the source YAML untouched.
- `report_market_regime.py` generates a read-only market regime and cash-ratio
  report from local merged Parquet data. It does not connect to broker software,
  place orders, or modify Qlib provider files.
- `report_cash_control_backtest.py` reads the latest Qlib daily portfolio
  backtest report and applies the market-regime cash ratio as a read-only
  overlay. It compares baseline versus cash-control metrics, writes
  `cash_control_backtest_latest.{json,md}`, and does not connect to broker
  software, place orders, or modify Qlib provider files.
- `report_qlib_predictions.py` reads the latest Qlib/MLflow `pred.pkl`
  artifact and writes `qlib_predictions_latest.{json,md}` with latest-date
  top/bottom model scores. It is a read-only ranking summary, not an automatic
  trading signal, and it does not connect to broker software, place orders, or
  modify Qlib provider files.
- `optimize_strategy.py --dry-run --skip-training` runs a browser-safe
  read-only smoke experiment for low-turnover, market-timing, and drawdown
  protection logic. It writes `optimization_results_latest.{json,md}` and
  does not train, connect to broker software, place orders, or modify Qlib
  provider files. Run the non-dry-run path manually only when you explicitly
  want a longer local experiment.

Research-only model comparison scripts live under:

`quant_workflow/scratch/experiments/model_research`

Do not promote a script from that folder into `scripts/` until its assumptions,
data splits, and output paths are documented here.

The current main provider defaults to:

`${QUANT_DATA_ROOT}/qlib/cn_data_baostock_jqdata`

The optional cleaned provider defaults to:

`${QUANT_DATA_ROOT}/qlib/cn_data_baostock_jqdata_cleaned`
