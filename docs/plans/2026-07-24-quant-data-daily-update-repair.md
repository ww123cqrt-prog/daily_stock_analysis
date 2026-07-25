# Quant Data Daily Update Repair Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the scheduled DSA `quant_data` refresh fail closed, publish cleaned data atomically, and leave complete evidence for every trigger.

**Architecture:** Downloads remain independent source caches. A run that has enough current-market coverage builds merged and cleaned artifacts under a run-scoped staging directory, validates those candidates, then promotes all four derived artifacts and synchronizes the manifest in one rollback-capable publish operation. Runtime status and a process-aware lock are written outside the official data directories.

**Tech Stack:** Python 3.11, pandas, NumPy, Qlib, pytest/unittest, macOS launchd.

---

### Task 1: Runtime lock and status evidence

**Files:**
- Create: `quant_workflow/scripts/_daily_update_runtime.py`
- Modify: `quant_workflow/scripts/run_daily_update.py`
- Test: `quant_workflow/tests/test_daily_update_runtime.py`
- Test: `quant_workflow/tests/test_run_daily_update.py`

**Steps:**

1. Write failing tests for an active PID lock, a dead/stale lock, owned release, and a duplicate trigger recorded as `controlled_skip`.
2. Run the focused tests and confirm failures are caused by the absent runtime module/behavior.
3. Implement atomic JSON status writes plus an exclusive JSON lock containing PID, run ID, start time, and host; only reclaim locks whose process is dead or whose malformed lock has expired.
4. Acquire the lock before any pipeline work and release it in `finally`; update runtime status at start, each stage boundary, controlled skip, failure, and success.
5. Re-run focused tests.

### Task 2: Market coverage gate and downloader status

**Files:**
- Modify: `quant_workflow/scripts/_ashare_sources.py`
- Modify: `quant_workflow/scripts/run_daily_update.py`
- Test: `quant_workflow/tests/test_ashare_sources.py`
- Test: `quant_workflow/tests/test_run_daily_update.py`

**Steps:**

1. Write failing tests proving that one updated symbol cannot advance the market date, one sufficiently covered source can proceed while two fail, and all failed sources return nonzero when no rebuild occurs.
2. Add a full-file coverage profile keyed by the candidate latest date and the configured stock universe; production default requires a documented minimum ratio.
3. Persist every downloader summary and failure sample under `reports/runtime/status` while retaining existing `reports/download_state` compatibility files.
4. Attach source summaries and coverage to `daily_update_summary_latest.json` and fail with `source_coverage_gate` when the newest date has insufficient coverage.
5. Re-run focused tests.

### Task 3: Benchmark fallback without fabricated prices

**Files:**
- Modify: `quant_workflow/scripts/add_index_benchmark.py`
- Test: `quant_workflow/tests/test_add_index_benchmark.py`

**Steps:**

1. Write failing tests for Baostock failure followed by a valid EastMoney fallback, a validated local Qlib cache fallback, incomplete-source rejection, and total-source fail-closed behavior.
2. Implement ordered anonymous sources: Baostock, EastMoney, AkShare, then local benchmark-provider caches.
3. Validate that every target provider calendar day has a finite positive close; never forward-fill a missing date.
4. Write benchmark features only after source validation and persist a sanitized benchmark source/attempt status file.
5. Re-run focused tests.

### Task 4: Calendar-safe benchmark alignment

**Files:**
- Modify: `quant_workflow/scripts/apply_data_cleaning.py`
- Test: `quant_workflow/tests/test_apply_data_cleaning.py`

**Steps:**

1. Write failing tests where the source calendar has an extra day and where the target calendar has an unsupported extra day.
2. Decode each Qlib feature bin against its source calendar, map by date, and rewrite it against the target calendar.
3. Permit source-only dates, reject target dates that have no source benchmark observation, and keep the benchmark out of tradable instrument files.
4. Re-run focused tests.

### Task 5: Candidate validation and atomic official publish

**Files:**
- Create: `quant_workflow/scripts/publish_daily_candidate.py`
- Modify: `quant_workflow/scripts/run_daily_update.py`
- Modify: `quant_workflow/scripts/report_data_quality.py`
- Modify: `quant_workflow/scripts/check_qlib_data.py`
- Modify: `quant_workflow/scripts/sync_data_manifest.py`
- Test: `quant_workflow/tests/test_publish_daily_candidate.py`
- Test: `quant_workflow/tests/test_report_data_quality.py`
- Test: `quant_workflow/tests/test_check_qlib_data.py`
- Test: `quant_workflow/tests/test_sync_data_manifest.py`

**Steps:**

1. Write failing tests proving failed cleaning/quality/manifest leaves the previous official provider and manifest unchanged.
2. Pass run-scoped staging paths through merge, benchmark, snapshot, anomaly, cleaning, quality, and Qlib checks.
3. Make quality inspect the cleaned candidate, validate complete benchmark coverage, and return nonzero when the production readiness gate fails.
4. Promote merged parquet, merged Qlib, cleaned parquet, and cleaned Qlib with same-filesystem renames; preserve the previous directories as run-scoped rollback artifacts.
5. Synchronize the manifest inside the publish transaction; rollback all promoted artifacts if manifest generation fails.
6. Merge DSA-owned manifest fields into the existing manifest so robust-strategy entries survive while stale `daily_stock_analysis-robust` report links migrate to the official DSA reports root.
7. Re-run focused tests.

### Task 6: Isolated end-to-end and launchd verification

**Files:**
- Modify: `quant_workflow/README.md`
- Verify: `/Users/cq/Library/LaunchAgents/com.dsa.quant-daily-update.plist`

**Steps:**

1. Document `report_data_hygiene.py` as manual read-only governance reporting, never an automatic delete/move task.
2. Run all relevant tests and `compileall` using `.venv/bin/python`.
3. Build a tiny temporary `QUANT_DATA_ROOT`, run the complete no-network pipeline through merge, benchmark cache, anomaly, cleaning, Qlib rebuild, quality, publish, and manifest, then prove the real shared data timestamps/hashes were unchanged.
4. Recheck robust training; do not run the production pipeline while it is active.
5. Validate plist syntax and paths, exercise `bootout`/`bootstrap`, enable the service, and verify it is registered without kickstarting a production run.
6. Inspect final runtime evidence and report exact source, merged, cleaned, manifest, and launchd states; explicitly state that production full-market validation remains pending if robust training is still active.
