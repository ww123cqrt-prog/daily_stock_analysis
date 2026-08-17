# quant_data Increment Validation Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add and execute read-only old/new snapshot comparison and deterministic increment sampling for governed A-share snapshots.

**Architecture:** Put reusable validation logic in a focused data-layer module and expose it through two `quant_data_cli.py` subcommands. Resolve snapshots by governed dataset identity, compare complete historical semantics offline, and compare sampled new rows against the immutable local Baostock increment bundle.

**Tech Stack:** Python 3.11, pandas, PyArrow/Parquet, argparse, pytest, quant_data v2 manifests.

---

### Task 1: Snapshot comparison contract

**Files:**
- Create: `quant_workflow/scripts/quant_data_increment_validation.py`
- Test: `quant_workflow/tests/test_quant_data_increment_validation.py`

1. Write failing tests for valid raw/qfq increments, historical drift, symbol mismatch, missing dates, and unregistered factor-history changes.
2. Run the focused tests and confirm they fail because the module/API does not exist.
3. Implement governed snapshot resolution, index loading, historical comparison, factor-change exceptions, and JSON result construction.
4. Run the focused tests and confirm they pass.

### Task 2: Increment bundle sampling contract

**Files:**
- Modify: `quant_workflow/scripts/quant_data_increment_validation.py`
- Test: `quant_workflow/tests/test_quant_data_increment_validation.py`

1. Write failing tests for deterministic sampling, bundle hash/date/value checks, category coverage, and tampered bundle rows.
2. Run the focused tests and confirm expected failures.
3. Implement bounded deterministic sampling and offline bundle comparison. Sampling must rerun the complete semantic comparison and require the supplied comparison report to match the live snapshot result exactly; qfq lineage binds through the selected qfq base snapshot's unique raw upstream identity.
4. Run the focused tests and confirm they pass.

### Task 3: CLI integration

**Files:**
- Modify: `quant_workflow/scripts/quant_data_cli.py`
- Modify: `quant_workflow/tests/test_quant_data_cli.py`

1. Write failing CLI tests for `compare-snapshots` and `sample-increment`, JSON output files, and nonzero failure status.
2. Add parser definitions and minimal command dispatch using the shared validator.
3. Run the CLI tests and confirm they pass.

### Task 4: Documentation and execution

**Files:**
- Modify: `quant_workflow/docs/quant-data-contract.md`
- Modify: `docs/CHANGELOG.md`

1. Document both commands, result boundaries, and the offline bundle truth source.
2. Run Python compilation and all affected unit tests.
3. Run complete snapshot comparison for raw and qfq from the `2026-08-04` snapshots to the `2026-08-14` snapshots.
4. Run deterministic increment sampling against the retained local Baostock bundle.
5. Save reports under `quant_workflow/reports/quant_data_validation/` and report exact checked/unchecked counts.
