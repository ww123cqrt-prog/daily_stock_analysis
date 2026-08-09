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
  `QLIB_CLEANED_PROVIDER_URI`, `QLIB_COMPATIBILITY_PROVIDER_URI`,
  `QLIB_CLEANED_COMPATIBILITY_PROVIDER_URI`, `QLIB_DIRECT_MEMBERSHIP_PATH`,
  `QLIB_DIRECT_CLEANING_PLAN_PATH`, and `CHINA_ASTOCK_QUANT_ROOT`
  override specific
  paths when needed.
- Do not add new hardcoded local absolute paths in scripts.

Naming convention:

- `download_<source>.py` fetches source data into Parquet.
  `download_jqdata.py --dry-run` is the safe authentication and stock-pool
  parsing check used by the web workflow; omit `--dry-run` only when you intend
  to write real Parquet files.
- `supervise_<source>.py` runs a restartable long download.
- `build_qlib_direct_provider.py` is the formal Qlib entry. It creates a
  metadata-only descriptor that reads the governed Parquet snapshot directly;
  the canonical and cleaned modes use the same stock/index data and differ
  only by the frozen cleaning plan.
- `merge_to_qlib.py`, `build_qlib_v2.py` and `build_cleaned_qlib_v2.py` are
  compatibility/audit producers. They may create `.bin` data only when an
  operator explicitly runs the old tool; they are not the DSA web default and
  are not used by training, prediction or backtest callers.
- `apply_data_cleaning.py` remains a bounded read-only/dry-run legacy report
  tool. The normal web action now creates the cleaned direct descriptor with
  `build_qlib_direct_provider.py --cleaning-mode cleaned`.
- `check_qlib_data.py` smoke-checks the resolved direct Provider by default.
  `--provider cleaned` performs the same real Qlib read through the frozen
  cleaned descriptor.
- `quant_data_cli.py update cn_index_daily --through YYYY-MM-DD` is the formal
  five-index incremental update. It requests only SH000300, SH000905,
  SH000906, SZ399006 and SH000688, verifies the bounded overlap, resolves the
  requested calendar date to the latest XSHG session, and atomically publishes
  `cn_index_daily`. It never updates stocks or creates Qlib `.bin` files.
- `update_qlib_index_benchmarks.py --publish` is a legacy native-provider index
  supplement retained only for an explicitly identified old consumer. The
  formal five-index set is `cn_index_daily`, which the direct Provider reads
  directly; normal updates do not publish this native `.bin` supplement.
- `compare_native_qlib_benchmarks.py` joins SH000300/SH000905/SH000906 by
  trading date and requires exact equality for open/high/low/close/volume/factor
  across the full retained compatibility overlap. Directory existence is not
  treated as evidence.
- `repair_qlib_provider.py` fixes provider instrument metadata without a full rebuild.
- `add_index_benchmark.py` and `generate_csi300.py` are provider-maintenance
  scripts. They now support `--dry-run` and parameterized paths, but should
  stay at the script layer until they write report artifacts suitable for the
  DSA web UI.
- `supervise_baostock.py` is a restartable long-download supervisor. It is not
  exposed as a separate DSA web button because the web task center already wraps
  Baostock downloads with durable job logs and cancellation.
- `report_data_quality.py` scans native `ashare_daily_qfq` bars and the direct
  canonical/cleaned descriptors by default, while retaining legacy-source
  overlap evidence.
  Use `--overlap-sample` to limit
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
- `run_candidate_pipeline.py` runs the fixed five-model Alpha158 comparison
  (`Linear OLS`, `Linear Ridge`, `LightGBM`, `XGBoost`, `CatBoost`). It selects
  two models and their five strategy overlays only on 2024-2025 data, then
  reports a separate 2026 holdout check. Every run is recorded in MLflow, but
  the script restores the pre-existing `latest_training_result.json` and never
  promotes a candidate or places an order automatically. Use `--dry-run` for a
  deterministic small-sample pipeline check.
- `report_market_regime.py` generates a read-only market regime and cash-ratio
  report from resolved native qfq bars. It does not connect to broker software,
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
- `quant_data_cli.py` is the governed entrypoint for `list`, `describe`,
  `resolve`, `validate`, and `update`. `resolve` checks status, requested price
  basis, manifest hash, bar hashes, factor hashes and company-action hash.
- `build_qlib_direct_provider.py` creates the metadata-only Qlib adapter for the
  current governed `ashare_daily_qfq` snapshot and runs a real `D.features`
  smoke check. Without an explicit date range, it deterministically samples up
  to five dates from each symbol's own Parquet history, so a stock suspension
  cannot be mistaken for a Provider failure. Every requested field must be
  present and non-null on each sampled row. It writes only `CALENDAR.parquet`, a frozen
  `MEMBERSHIP_INTERVALS.parquet`, and `PROVIDER_DESCRIPTOR.json`; it never
  creates `features/` or stock `.bin` files. Python callers initialize it
  through `_paths.init_qlib_direct()`.
- `quant_data_cli.py update cn_etf_daily` freezes the current 51-symbol universe,
  reuses its existing files, and requests only a bounded overlap plus missing
  dates. An unchanged target date returns with zero provider requests. If the
  overlap proves one symbol's provider adjustment changed, only that symbol may
  receive a targeted history refresh. `download_cn_etf_data.py` retains the
  separate full-rebuild producer, but its command line refuses to run unless
  severe current corruption, an unrecoverable backup, and explicit user
  approval are all confirmed.
  Tencent raw/qfq history uses a single-instance lock, request budget and
  bounded retries.
  Paired adjustment evidence uses piecewise affine transforms
  (`qfq = raw * factor + offset`) so cash distributions are not mistaken for
  multiplicative splits. One immutable snapshot contains `raw/`, `qfq/`,
  `factors/` and one manifest. Incomplete runs remain in cache and never update
  `LATEST.json`. After publication, retention and root-manifest housekeeping
  normally leave only the current and immediately previous complete snapshots;
  a housekeeping failure returns `published_with_warnings` without pretending
  that the already-committed snapshot was not published.
  `download_etf_momentum_data.py` now delegates its
  command-line publication to this unified producer and cannot create a second
  formal ETF dataset.
- `quant_data_cli.py validate cn_etf_daily --release` is the single offline ETF
  release check. It validates the current snapshot and sole backup, all
  raw/qfq/factor files, corporate actions, the frozen universe, root catalog,
  retired ETF paths, and the shared qfq availability of the five registered
  strategy symbols (`SH512890`, `SH513010`, `SH510720`, `SZ159915`, and
  `SH513100`). It does not scan unrelated A-share datasets. The daily updater
  runs it automatically after the ETF update and before report generation. If
  either the update or release check fails, the previous report is retained and
  the unrelated A-share stages continue.
- `download_baostock_v2.py` creates a restartable full-market Baostock source
  bundle containing complete raw bars and `query_adjust_factor` events. It
  rebuilds qfq locally as raw OHLC times `foreAdjustFactor`, while provider qfq
  is retained only for recent-event, bounded factor-start-gap, and
  deterministic-sample verification. For long factor-history gaps caused by
  code restructuring, it derives the missing daily factor only from a complete
  same-provider raw/qfq OHLC pair and records the derived transitions. Before
  publication it audits all overlap
  against legacy Baostock qfq, separating constant renormalization and proven
  legacy incremental splices from unexplained price-path drift.
  `quant_data_cli.py update ashare_daily_qfq --bundle-dir <verified-bundle>`
  validates all hashes and switches raw/qfq snapshots plus the root catalog as
  one rollback transaction, then builds canonical and cleaned direct Provider
  descriptors before the formal update is considered complete. The direct
  descriptors contain no copied stock market data.
- `build_qlib_v2.py` is retained as an explicit compatibility/audit producer.
  It still creates a physical `.bin` copy and is not called by the normal update,
  training, prediction or backtest path.
- `build_cleaned_qlib_v2.py` is the matching explicit legacy cleaned producer.
  The formal cleaned Provider instead reuses the qfq Parquet snapshot and
  applies its frozen cleaning plan at read time.
- `build_robust_price_provider.py` is a separate legacy robust-strategy product
  builder. It may create `.bin` files only inside its hash-isolated compatibility
  product; the live robust pipeline uses `prepare_robust_provider.py` over the
  formal direct descriptor and does not call this builder.
- `publish_v2_compatibility.py` rebuilds the four stable V1 Parquet/legacy-Qlib paths
  from the exact governed qfq snapshot. It preserves six-digit Parquet names,
  nanosecond dates, the V1 column order, and unit Qlib factors because prices
  are already qfq. It runs a fresh anomaly/cleaning pass, installs Tencent
  CSI300/CSI500/CSI800 benchmark features, validates all four candidates, then
  promotes them together under an exclusive publication lock. Candidate
  publication also verifies the exact `stock_lists.json` hash used to build
  CSI300. Tencent benchmark requests are rate-limited and use bounded retry;
  one missing benchmark rejects the full candidate. On the first promotion,
  previous V1 outputs are retained under `_old` for rollback and audit.
  Normal V2 updates do not call this script. Compatibility publication occurs
  only when `quant_data_cli.py update ashare_daily_qfq` is explicitly given
  `--refresh-legacy-compatibility`.
- `sync_data_manifest.py` writes schema-v2 `DATA_MANIFEST.json`, keeps all data
  paths relative to `QUANT_DATA_ROOT`, refreshes generated README catalog
  blocks, and supports `--check` for catalog/path/README drift.
- `generate_etf_momentum_strategy_report.py` is the sole calculation and report
  entry for the fixed two-ETF 20-day ranking and moving-average rule. It writes
  the core backtest and manual-review robustness evidence into one versioned
  report under `reports/etf_momentum_rotation/`, including cost cases,
  one-session execution delay, pre-declared parameter sensitivity, explicit
  `CASH` periods and price-discontinuity evidence. The result is a research
  candidate, not a promoted model or an order instruction.
- `backtest_etf_momentum_rotation.py` and
  `evaluate_etf_momentum_robustness.py` are compatibility wrappers around that
  generator; they do not publish separate formal reports or choose a
  parameter-grid winner.
- `research_etf_momentum_improvement.py` applies the user-supplied single-side
  commission from `configs/etf_momentum_improvement.yaml`, selects one bounded
  15/20/25-day candidate from the 2019-2022 development split only, and reports
  2023-2024 validation plus later observations separately. Outputs stay under
  `reports/etf_momentum_rotation/improvement/`; the original 20/20 rule and its
  reports are not overwritten. Because the grid was visible in earlier work,
  any selected rule remains a retrospective research candidate.

Research-only model comparison scripts live under:

`quant_workflow/scratch/experiments/model_research`

Do not promote a script from that folder into `scripts/` until its assumptions,
data splits, and output paths are documented here.

The formal storage path is `quant_data_qlib_provider.py`: it freezes the current
`ashare_daily_qfq` and `cn_index_daily` identities and reads their Parquet bars
directly. All training, prediction, backtest, calendar and data-quality
callers use `_paths.init_qlib_workflow()` / `qlib_workflow_provider_uri()` so
Qlib receives the custom calendar, instrument and feature providers. A direct
descriptor records the exact stock snapshot, index snapshot, membership hash
and (for cleaned mode) cleaning-plan hash; changing `LATEST.json` later cannot
change an already-running process.

The cleaned Provider is not a second market-data copy. It points to the same
`ashare_daily_qfq` Parquet snapshot and applies the frozen cleaning plan while
reading. `QLIB_PROVIDER_URI` and `QLIB_CLEANED_PROVIDER_URI` are retained only
for explicitly pinned legacy native replays and are rejected by the direct
default resolver; compatibility paths use the explicit
`QLIB_COMPATIBILITY_PROVIDER_URI` names.

Mutable maintenance scripts such as `merge_to_qlib.py`,
`add_index_benchmark.py`, `generate_csi300.py` and
`repair_qlib_provider.py` remain compatibility/audit tools. They do not change
the formal Provider and are not invoked by the paused daily update. Qlib config
files are runtime templates; DSA wrappers replace the provider configuration
with the frozen direct descriptor before invoking Qlib.

`run_robust_strategy.py` also fails closed when the formal direct Provider
cannot be built. Its `--allow-legacy-compatibility` flag is only for an
explicit replay of an old frozen run; it is not a normal live-run fallback.

## Robust Signal Goal 1

The first robust-research goal is isolated from the legacy full-strategy run.
Formal runs must use the launchd wrapper, a clean Git worktree, and the shared
quant data root resolved by `_paths.quant_data_root()`:

```bash
# Start a new durable formal run
.venv/bin/python quant_workflow/scripts/launch_robust_signal_research.py \
  --run-id robust_signal_YYYYMMDD_HHMMSS

# Pause after the currently running atomic job finishes
.venv/bin/python quant_workflow/scripts/launch_robust_signal_research.py \
  --run-id robust_signal_YYYYMMDD_HHMMSS --request-pause

# Resume only the same run and frozen context
.venv/bin/python quant_workflow/scripts/launch_robust_signal_research.py \
  --run-id robust_signal_YYYYMMDD_HHMMSS --resume
```

The formal report namespace is `reports/robust_signal/`, and the MLflow
experiment is `robust_csi800_signal_research`. A smoke run validates wiring
only; it is not signal evidence. Goal 1 rejects prediction rows dated 2026 or
later. If price-volume signals fail, `signal_diagnosis_complete` means the
conditional historical-fundamental stage is still required; it does not mean
`signal_not_found`.

Each completed job writes two factor-quality views: `raw` is the Alpha158
handler output before infer processors and exposes real missing/non-finite
values; `processed` is the model input after robust normalization, fillna,
board flags, and four point-in-time CSI800 market-regime one-hot features.
Market regimes are calculated continuously from the frozen `2014-01-02`
history start through each job's test end. The runner requires at least 60
trading days of warm-up before training and never changes the regime-history
start with the 504/756/1008-day training window. Attribution uses one
global horizon schedule before splitting dates by regime, so the state periods
sum back to the total evaluation schedule.
