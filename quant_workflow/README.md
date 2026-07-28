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
china-astock-quant adapters, and every stable script here. Data paths use
`QUANT_DATA_ROOT` when configured; otherwise the resolver uses the DSA-local
`quant_data/` directory when present and then the sibling WorkingProjects
`quant_data/` directory. Do not construct this path from the workflow directory.

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
# Build a governed A-share raw/factor/qfq source bundle (normally on a durable remote runner)
.venv/bin/python quant_workflow/scripts/download_baostock_v2.py download \
  --universe-file /path/to/stock_lists.json \
  --output-dir /path/to/staging-bundle \
  --through YYYY-MM-DD --resume

# Publish verified A-share v2 raw/qfq, Qlib v2, and V1-compatible outputs
.venv/bin/python quant_workflow/scripts/quant_data_cli.py update \
  ashare_daily_qfq --bundle-dir /path/to/verified-bundle

# Build and publish Qlib from the exact governed qfq snapshot
.venv/bin/python quant_workflow/scripts/build_qlib_v2.py --publish

# Supervise Baostock in restartable batches
.venv/bin/python quant_workflow/scripts/supervise_baostock.py

# Rebuild mutable V1-compatible merged output (not canonical V2)
.venv/bin/python quant_workflow/scripts/merge_to_qlib.py

# Repair provider metadata after manual benchmark/instrument edits
.venv/bin/python quant_workflow/scripts/repair_qlib_provider.py

# Smoke-check the resolved native Qlib V2 provider
.venv/bin/python quant_workflow/scripts/check_qlib_data.py

# Apply anomaly cleaning rules into a separate cleaned provider
.venv/bin/python quant_workflow/scripts/apply_data_cleaning.py

# Update independent sources and validate existing native A-share V2. A history
# source selected here updates audit cache only; canonical publication requires
# --ashare-v2-bundle-dir /path/to/verified-bundle on the command line.
.venv/bin/python quant_workflow/scripts/run_daily_update.py

# Read-only data-governance inventory and dry-run cleanup plan
.venv/bin/python quant_workflow/scripts/report_data_hygiene.py
# Refresh and atomically publish the governed two-ETF raw/qfq snapshots
.venv/bin/python quant_workflow/scripts/quant_data_cli.py update etf_momentum_daily_qfq

# Inspect data basis, adjustment evidence, date range and lineage
.venv/bin/python quant_workflow/scripts/quant_data_cli.py describe etf_momentum_daily_qfq

# Fail when the catalog, governed paths or generated README tables drift
.venv/bin/python quant_workflow/scripts/sync_data_manifest.py --check

# Run the canonical Qlib Alpha158 + LightGBM baseline
.venv/bin/python quant_workflow/scripts/train_baseline_lgb.py

# Run the same baseline on the cleaned provider
.venv/bin/python quant_workflow/scripts/train_baseline_lgb.py --provider cleaned --experiment-name baseline_lgb_alpha158_cleaned
```

## Current Data Outputs

- Canonical A-share Qlib v2 snapshots: `quant_data/qlib/cn_data_ashare_v2/snapshots/<snapshot_id>`
- Baostock Parquet: `quant_data/parquet/ashare_1d/baostock_qfq`
- JQData Parquet: `quant_data/parquet/ashare_1d/jqdata`
- V1-compatible merged Parquet generated from V2: `quant_data/parquet/ashare_1d/merged_baostock_jqdata`
- V1-compatible Qlib provider generated from V2: `quant_data/qlib/cn_data_baostock_jqdata`
- V1-compatible cleaned Parquet generated from V2: `quant_data/parquet/ashare_1d/merged_baostock_jqdata_cleaned`
- V1-compatible cleaned Qlib generated from V2: `quant_data/qlib/cn_data_baostock_jqdata_cleaned`
- Pre-migration rollback/audit copies: the same four names with `_old` suffix
- Canonical global daily Parquet: `quant_data/parquet/global_1d/{us,jp,kr}`
- Yahoo raw responses: `quant_data/raw/global_1d/yahoo/captured_date=YYYY-MM-DD`
- Sina snapshot raw responses: `quant_data/raw/ashare_snapshots/captured_date=YYYY-MM-DD`
- EastMoney news raw responses: `quant_data/raw/ashare_news/captured_date=YYYY-MM-DD`
- Governed ETF raw evidence: `quant_data/parquet/etf_1d/etf_momentum_daily_raw`
- Governed ETF qfq snapshot: `quant_data/parquet/etf_1d/etf_momentum_daily_qfq`

The canonical storage, schema, merge, and handoff rules are documented in
[`docs/quant-data-contract.md`](docs/quant-data-contract.md). The live local
paths and readiness state remain authoritative in
`${QUANT_DATA_ROOT}/DATA_MANIFEST.json`; a robust run must also pass
`preflight_robust_data.py`.

The governed A-share v2 entries are `ashare_daily_raw`, `ashare_daily_qfq`, and
`qlib_ashare_v2`; resolve them through `quant_data_cli.py` instead of hard-coding
snapshot paths. DSA canonical training, prediction, backtesting, trading-calendar
and quality checks resolve `qlib_ashare_v2` once per process. The original
merged paths remain generated compatibility outputs for rollback, mutable
maintenance tools and callers that still require cleaned semantics; they are
not the canonical DSA input. Only the `_old` directories and old source
directories are audit-only. Do not infer readiness from directory names; the
live manifest is authoritative.

The robust strategy configuration uses verified offline reuse for the one-time
reference snapshot and the three Qlib benchmarks. It does not silently redownload
when local data is missing or mismatched; the correct state is `blocked_data`.
Remote artifacts remain staging until they are imported locally with
`download_robust_reference.py --import-dir` and
`add_index_benchmark.py --import-provider-dir`. Both commands revalidate local
identity and use recoverable transactions; direct copies into canonical data
directories are not supported.

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
  provider for request-load purposes. These source jobs update audit caches
  only and can never publish canonical V2 data.
- A canonical A-share daily publication requires
  `--ashare-v2-bundle-dir <verified-bundle>`. The runner delegates that bundle
  to `quant_data_cli.py`; the web whitelist intentionally does not accept an
  arbitrary local bundle path. Without a bundle, the A-share portion is a
  validation-only run.
- `train_baseline_lgb.py --provider cleaned` rewrites a runtime YAML under
  `reports/runtime_configs/` so the source config stays canonical.
- Qlib YAML files under `configs/` contain the
  `RESOLVE_QLIB_ASHARE_V2_AT_RUNTIME` sentinel. Run them through the DSA
  wrappers, which replace the sentinel with one immutable snapshot path before
  Qlib starts. Direct `qrun` use of a source template is unsupported.
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
- The daily runner never invokes the legacy merged/cleaned candidate publisher.
  V2 bundle publication follows immutable staging, validation and pointer
  rollback inside `quant_data_cli.py`; fixed-path compatibility outputs are
  maintained by `publish_v2_compatibility.py` only.
- Benchmark fallback is fail-closed. Baostock, anonymous public-source
  fallbacks, and validated local provider caches are tried in order; missing
  benchmark dates are never silently forward-filled as real closes.
- `report_data_hygiene.py` intentionally remains a manual, read-only governance
  report rather than part of the weekday 18:40 chain. It inventories data and
  writes a dry-run cleanup plan only. It never deletes or moves data, and any
  future low-frequency schedule must preserve that read-only contract.
