# Known Issues

## Training Failures

### Cleaned compatibility benchmark drift (2026-07-04)

**Issue**: `train_baseline_lgb` fails with error:
```
ValueError: The benchmark ['SH000300'] does not exist. Please provide the right benchmark
```

**Root Cause**: An older cleaned compatibility provider was published without a complete CSI 300 benchmark.

**Impact**: A run explicitly selecting that stale cleaned provider cannot complete benchmark backtesting.

**Resolution**: Canonical DSA training, prediction and backtesting now resolve the immutable `ashare_daily_qfq` Parquet snapshot through the direct Provider. The formal `cn_index_daily` snapshot must contain SH000300, SH000905, SH000906, SZ399006 and SH000688 before the Provider is usable; the updater also validates the required fields through the frozen Provider calendar end.

**Status**: Fixed for the canonical direct Provider. A deliberately selected cleaned view uses the same qfq Parquet snapshot plus a hash-locked cleaning plan and must pass its own quality check.

**Related Files**:
- Failed job: `qj_20260704134523_d891c362`
- Log: `quant_workflow/reports/jobs/<job_id>/job.log`
- Script: `quant_workflow/scripts/train_baseline_lgb.py`
- Config: `quant_workflow/configs/baseline_lgb_alpha158.yaml`

**Operational rule**: Do not disable the benchmark or substitute another index to make a failed run pass. Validate the formal index snapshot and direct Provider from the same governed qfq snapshot. The old cleaned `.bin` output is only a compatibility/audit artifact.
