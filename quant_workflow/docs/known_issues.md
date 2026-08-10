# Known Issues

## Training Failures

### Cleaned compatibility benchmark drift (2026-07-04)

**Issue**: `train_baseline_lgb` fails with error:
```
ValueError: The benchmark ['SH000300'] does not exist. Please provide the right benchmark
```

**Root Cause**: An older cleaned compatibility provider was published without a complete CSI 300 benchmark.

**Impact**: A run explicitly selecting that stale cleaned provider cannot complete benchmark backtesting.

**Resolution**: Canonical DSA training, prediction and backtesting now resolve the immutable `qlib_ashare_v2` snapshot, which must contain SH000300, SH000905 and SH000906 before publication. The updater also validates all six required fields through the provider calendar end.

**Status**: Fixed for the canonical native provider. A deliberately selected cleaned provider remains an independent data product and must pass its own quality check.

**Related Files**:
- Failed job: `qj_20260704134523_d891c362`
- Log: `quant_workflow/reports/jobs/<job_id>/job.log`
- Script: `quant_workflow/scripts/train_baseline_lgb.py`
- Config: `quant_workflow/configs/baseline_lgb_alpha158.yaml`

**Operational rule**: Do not disable the benchmark or substitute another index to make a failed run pass. Rebuild the cleaned compatibility output from the same governed qfq snapshot, or use the canonical native provider.
