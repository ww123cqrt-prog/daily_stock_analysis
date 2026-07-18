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

The canonical storage, schema, merge, and handoff rules are documented in
[`docs/quant-data-contract.md`](docs/quant-data-contract.md). The live local
paths and readiness state remain authoritative in
`${QUANT_DATA_ROOT}/DATA_MANIFEST.json`; a robust run must also pass
`preflight_robust_data.py`.

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
- `train_baseline_lgb.py --provider cleaned` rewrites a runtime YAML under
  `reports/runtime_configs/` so the source config stays canonical.
- `SH000300` benchmark data may exist under `features/sh000300`, but it must
  not be listed in `instruments/all.txt` or `instruments/csi300.txt`; otherwise
  model training can treat the index as a tradable stock.
