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

The path resolver keeps a transitional fallback for older installations: if the
embedded runtime folders have no history markers but a sibling `quant_workflow`
does, DSA can still read the sibling data. Once embedded markers exist, embedded
reports and MLflow runs win. This prevents a restarted DSA service from showing
an empty `/research` page while keeping the repo-local layout as the source of
truth.
