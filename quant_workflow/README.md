# Quant Workflow

This folder keeps the local A-share data workflow inside the DSA repository,
outside the cloned Qlib source tree.

## Layout

- `scripts/` - durable scripts we expect to reuse.
- `configs/` - Qlib workflow YAML configs. The canonical baseline is
  `baseline_lgb_alpha158.yaml`.
- `experimental/` - importable research-only handlers/strategies. These are
  not part of the canonical baseline until separately validated.
- `reports/` - local data quality reports, experiment notes, jobs and MLflow
  summaries. This directory is runtime output and is ignored by Git.
- `scratch/` - local temporary experiments. Only `scratch/README.md` is kept in
  the repository.

## Durable Scripts

Run commands from the DSA repository root. Python runtime is shared with the
DSA backend: use `.venv/bin/python` for Qlib, JQData/AkShare,
china-astock-quant adapters, and every stable script here. Data paths default
to `quant_data/` under the DSA repository; set `QUANT_DATA_ROOT` when the local
Parquet or Qlib provider root moves.

```bash
# Verify JQData auth and stock-pool parsing without writing data
.venv/bin/python quant_workflow/scripts/download_jqdata.py --universe csi300 --limit 3 --dry-run

# Real JQData download, writes Parquet files under quant_data
.venv/bin/python quant_workflow/scripts/download_jqdata.py --universe all

# Download Baostock long-history data
.venv/bin/python quant_workflow/scripts/download_baostock.py --retry-failed

# Download the CatDesk US/Japan/Korea universe and retain each raw Yahoo response
.venv/bin/python quant_workflow/scripts/download_global_daily.py --incremental

# Download a bounded A-share news sample and retain each raw EastMoney response
.venv/bin/python quant_workflow/scripts/download_ashare_news.py --limit 50

# Supervise Baostock in restartable batches
.venv/bin/python quant_workflow/scripts/supervise_baostock.py

# Merge Baostock + JQData and convert to Qlib bin
.venv/bin/python quant_workflow/scripts/merge_to_qlib.py

# Repair provider metadata after manual benchmark/instrument edits
.venv/bin/python quant_workflow/scripts/repair_qlib_provider.py

# Smoke-check the converted Qlib provider
.venv/bin/python quant_workflow/scripts/check_qlib_data.py

# Apply anomaly cleaning rules into a separate cleaned provider
.venv/bin/python quant_workflow/scripts/apply_data_cleaning.py

# Run the chain without Baostock/EastMoney/AkShare history requests. A history
# source must be selected explicitly after provider access has been reviewed.
.venv/bin/python quant_workflow/scripts/run_daily_update.py

# Read-only data-governance inventory and dry-run cleanup plan
.venv/bin/python quant_workflow/scripts/report_data_hygiene.py

# Run the canonical Qlib Alpha158 + LightGBM baseline
.venv/bin/python quant_workflow/scripts/train_baseline_lgb.py

# Run the same baseline on the cleaned provider
.venv/bin/python quant_workflow/scripts/train_baseline_lgb.py --provider cleaned --experiment-name baseline_lgb_alpha158_cleaned
```

## Current Data Outputs

- Baostock Parquet: `quant_data/parquet/ashare_1d/baostock_qfq`
- JQData Parquet: `quant_data/parquet/ashare_1d/jqdata`
- Merged Parquet: `quant_data/parquet/ashare_1d/merged_baostock_jqdata`
- Qlib provider: `quant_data/qlib/cn_data_baostock_jqdata`
- Cleaned merged Parquet: `quant_data/parquet/ashare_1d/merged_baostock_jqdata_cleaned`
- Cleaned Qlib provider: `quant_data/qlib/cn_data_baostock_jqdata_cleaned`
- Canonical global daily Parquet: `quant_data/parquet/global_1d/{us,jp,kr}`
- Yahoo raw responses: `quant_data/raw/global_1d/yahoo/captured_date=YYYY-MM-DD`
- Sina snapshot raw responses: `quant_data/raw/ashare_snapshots/captured_date=YYYY-MM-DD`
- EastMoney news raw responses: `quant_data/raw/ashare_news/captured_date=YYYY-MM-DD`

## Notes

- JQData credentials live in the local project `.env`; never print them in job
  logs or reports.
- The DSA `/workflow` JQData button is intentionally a browser-safe dry run:
  it authenticates, resolves the stock pool, samples at most 20 stocks, and does
  not create Parquet files. Use the script command above for real downloads.
- Historical one-off scratch scripts are intentionally not kept in the public
  repository. Promote useful experiments into `scripts/`, `configs/`, `tests/`
  and docs before committing them.
- `configs/global_daily_symbols.json` is the canonical global universe consumed
  by the downloader. Each symbol has one incrementally merged Parquet file;
  timestamped raw responses are append-only evidence and are never committed.
- The A-share news downloader is bounded by default. It refuses an unbounded
  full-market request unless the operator supplies an explicit symbols file.
- A-share source jobs use a conservative safety profile: at least 2 seconds
  between requests, exactly one attempt, at most 3 consecutive failures, a
  6000-request budget, and no more than 7 overlap days. Provider denial signals
  such as Baostock `10001011` or HTTP 403/429 stop the source run immediately.
  These bounds reduce load but cannot guarantee provider acceptance because the
  providers do not publish a complete rate-limit contract.
- `run_daily_update.py` defaults to `--ashare-source none` and `--news-limit 0`.
  Baostock, EastMoney, and AkShare are opt-in and mutually exclusive per run;
  AkShare's A-share history call uses EastMoney underneath and is not a separate
  provider for request-load purposes.
- Benchmark repair is also offline by default, so a missing local `SH000300`
  cache cannot silently reopen Baostock/EastMoney access. Only the explicit
  command-line flag `--allow-benchmark-network` enables those fallbacks; the
  scheduled task and DSA web whitelist do not expose that flag.
- `train_baseline_lgb.py --provider cleaned` rewrites a runtime YAML under
  `reports/runtime_configs/` so the source config stays canonical.
- `SH000300` benchmark data may exist under `features/sh000300`, but it must
  not be listed in `instruments/all.txt` or `instruments/csi300.txt`; otherwise
  model training can treat the index as a tradable stock.
- The daily updater uses `reports/runtime/status/daily_update.lock.json` for
  single-instance execution. A duplicate trigger is recorded as
  `controlled_skip`; a dead PID lock is reclaimed, while a live PID lock is
  never removed.
- DSA also permits only one active `quant_data` network job across the full
  daily update and individual download buttons, preventing web-triggered jobs
  from bypassing the daily updater lock.
- Baostock, EastMoney history, AkShare history, and EastMoney news scripts also
  share `source_download.lock.json`. Direct shell launches therefore cannot run
  two of these provider downloaders at the same time.
- Source parquet caches, run-scoped merged/cleaned candidates, and fixed-path
  official providers have separate roles. Candidates live under
  `quant_data/runtime/daily_update/runs/<run-id>/`; the four official derived
  directories and `DATA_MANIFEST.json` are promoted together only after every
  required check passes. Previous official directories remain under that
  run's `previous_official/` rollback directory.
- Benchmark fallback is fail-closed. Baostock, anonymous public-source
  fallbacks, and validated local provider caches are tried in order; missing
  benchmark dates are never silently forward-filled as real closes.
- `report_data_hygiene.py` intentionally remains a manual, read-only governance
  report rather than part of the weekday 18:40 chain. It inventories data and
  writes a dry-run cleanup plan only. It never deletes or moves data, and any
  future low-frequency schedule must preserve that read-only contract.
