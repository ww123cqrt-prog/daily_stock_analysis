# Quant Workflow Runtime Data

The canonical local runtime home for DSA quant research is this embedded
`daily_stock_analysis/quant_workflow/` directory.

Runtime outputs are intentionally ignored by Git:

- `quant_workflow/reports/` stores workflow jobs, quality reports, experiment
  summaries, latest pointers, and Markdown notebooks.
- `quant_workflow/mlruns/` stores the MLflow file store used by the local
  Qlib experiments.

Migration note:

- Historical reports may be copied from an older sibling
  `../quant_workflow/reports/` into this directory.
- Historical MLflow runs may be copied from an older sibling
  `../quant_workflow/mlruns/` into this directory.
- Any top-level sibling copy is now a rollback/deprecated copy. Do not write new
  DSA workflow output there unless you explicitly set `QUANT_WORKFLOW_ROOT`,
  `QUANT_WORKFLOW_REPORTS_ROOT`, or `QUANT_WORKFLOW_MLRUNS_DIR`.

The path resolver no longer auto-detects sibling history marker files. Embedded
reports and MLflow runs are the default source of truth. To reuse older sibling
runtime data, copy it into this directory, or explicitly set
`QUANT_WORKFLOW_REPORTS_ROOT` / `QUANT_WORKFLOW_MLRUNS_DIR` for a temporary
rollback.
