# Known Issues

## Training Failures

### Benchmark SH000300 Missing (2026-07-04)

**Issue**: `train_baseline_lgb` fails with error:
```
ValueError: The benchmark ['SH000300'] does not exist. Please provide the right benchmark
```

**Root Cause**: The configured Qlib provider, for example `quant_data/qlib/cn_data_baostock_jqdata_cleaned`, is missing CSI 300 index (SH000300) data.

**Impact**: Cannot run baseline LightGBM training with CSI 300 as benchmark.

**Workaround Options**:
1. Add CSI 300 index data to Qlib provider using `add_index_benchmark.py` script
2. Use different benchmark (e.g., SH000001 for Shanghai Composite Index)
3. Disable benchmark in training config (less recommended for production models)

**Status**: Documented, not fixed. Requires user decision on preferred benchmark source.

**Related Files**:
- Failed job: `qj_20260704134523_d891c362`
- Log: `quant_workflow/reports/jobs/<job_id>/job.log`
- Script: `quant_workflow/scripts/train_baseline_lgb.py`
- Config: `quant_workflow/configs/baseline_lgb_alpha158.yaml`

**Next Steps**:
1. Check if `add_index_benchmark.py` script exists and can fetch CSI 300 data
2. Verify benchmark data requirements in Qlib documentation
3. Offer user choice of benchmark or auto-add CSI 300 from available data source
