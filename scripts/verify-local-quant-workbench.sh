#!/usr/bin/env bash
# Local DSA quant workbench verification gate.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WEB_ROOT="${REPO_ROOT}/apps/dsa-web"
PYTHON_BIN="${DSA_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
QUANT_WORKFLOW_ROOT="${QUANT_WORKFLOW_ROOT:-${REPO_ROOT}/quant_workflow}"
LOCAL_BASE_URL="${DSA_WEB_E2E_BASE_URL:-http://127.0.0.1:8010}"
REQUIRED_QUANT_WORKFLOW_SCRIPTS=(
  prepare_baostock_refresh.py
  _ashare_sources.py
  download_baostock.py
  download_eastmoney.py
  download_akshare.py
  download_global_daily.py
  download_ashare_news.py
  download_jqdata.py
  merge_to_qlib.py
  validate_latest_snapshot.py
  apply_data_cleaning.py
  repair_qlib_provider.py
  generate_csi300.py
  add_index_benchmark.py
  check_qlib_data.py
  report_data_quality.py
  report_data_anomalies.py
  report_data_hygiene.py
  report_workbench_capabilities.py
  report_market_regime.py
  report_cash_control_backtest.py
  report_qlib_predictions.py
  report_qlib_research.py
  run_china_astock_comparison.py
  report_china_astock_indicators.py
  optimize_strategy.py
  train_baseline_lgb.py
  run_candidate_pipeline.py
  build_pit_csi800.py
  download_robust_reference.py
  validate_robust_price_bundle.py
  import_robust_price_bundle.py
  build_robust_price_provider.py
  preflight_robust_data.py
  prepare_robust_provider.py
  robust_backtest.py
  robust_dataset.py
  robust_evaluation.py
  robust_market_data.py
  robust_shadow_tracking.py
  robust_strategy_core.py
  run_robust_development.py
  run_robust_strategy.py
  run_robust_confirmation.py
  launch_robust_confirmation.py
  run_robust_shadow_day.py
  run_robust_shadow_schedule.py
  launch_robust_strategy.py
  report_robust_strategy_closure.py
  robust_signal_dataset.py
  robust_signal_evaluation.py
  robust_signal_models.py
  run_robust_signal_research.py
  launch_robust_signal_research.py
  run_daily_update.py
)

section() {
  printf '\n==> %s\n' "$1"
}

usage() {
  cat <<'EOF'
Usage: scripts/verify-local-quant-workbench.sh [phase]

Phases:
  quick            Focused backend/frontend checks for changed workbench code.
  backend          DSA-only backend gate (ci_gate.sh) plus full pytest suite.
  frontend         Full frontend vitest, lint, and build.
  quant-workflow   quant_workflow script tests using the DSA venv.
  browser-local    Browser smoke against the running local DSA service, default 8010.
  browser-fake     Managed fake-analysis browser smoke; does not call a real LLM.
  browser-hygiene  Data-governance browser smoke against the running local DSA service.
  opencode-review  Read-only opencode review of the current working tree, including untracked files.
  full             backend + frontend + quant-workflow + browser-hygiene + browser-fake + browser-local.
  complete         full + opencode-review.

Environment:
  DSA_PYTHON              Override Python interpreter; defaults to daily_stock_analysis/.venv/bin/python.
  QUANT_WORKFLOW_ROOT     Override quant_workflow path.
  DSA_WEB_E2E_BASE_URL    Override local browser target; defaults to http://127.0.0.1:8010.
  AGENT_BRIDGE            Override cli-agent-interop agent_bridge.py for opencode-review.
  OPENCODE_REVIEW_TIMEOUT_SECONDS
                          Timeout for opencode-review; defaults to 300 seconds.

Notes:
  browser-local, browser-hygiene, full, and complete require a running local DSA
  service at DSA_WEB_E2E_BASE_URL before this script starts.
  backend intentionally excludes quant_workflow; run quant-workflow, full, or
  complete when validating cross-project quant scripts.
  opencode-review and complete require cli-agent-interop's agent_bridge.py, or
  set AGENT_BRIDGE to an equivalent local bridge script.
EOF
}

ensure_python() {
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python interpreter not found or not executable: ${PYTHON_BIN}" >&2
    exit 2
  fi
}

ensure_npm() {
  if ! command -v npm >/dev/null 2>&1; then
    echo "npm is required for frontend/browser verification." >&2
    exit 2
  fi
}

ensure_npx() {
  if ! command -v npx >/dev/null 2>&1; then
    echo "npx is required for Playwright browser verification." >&2
    exit 2
  fi
}

ensure_local_dsa() {
  if ! curl -fsS "${LOCAL_BASE_URL}/api/v1/quant/status" >/dev/null; then
    echo "Local DSA service is not reachable at ${LOCAL_BASE_URL}." >&2
    echo "Start it before running browser-local or browser-hygiene." >&2
    exit 2
  fi
}

is_local_dsa_reachable() {
  curl -fsS "${LOCAL_BASE_URL}/api/v1/quant/status" >/dev/null
}

check_quant_workflow_scripts() {
  section "quant_workflow whitelisted script existence"
  if [[ ! -d "${QUANT_WORKFLOW_ROOT}/scripts" ]]; then
    echo "quant_workflow scripts directory not found: ${QUANT_WORKFLOW_ROOT}/scripts" >&2
    exit 2
  fi
  missing=()
  for script_name in "${REQUIRED_QUANT_WORKFLOW_SCRIPTS[@]}"; do
    if [[ ! -f "${QUANT_WORKFLOW_ROOT}/scripts/${script_name}" ]]; then
      missing+=("${script_name}")
    fi
  done
  if [[ ${#missing[@]} -gt 0 ]]; then
    printf 'Missing quant_workflow script(s): %s\n' "${missing[*]}" >&2
    exit 2
  fi
  if [[ ! -f "${QUANT_WORKFLOW_ROOT}/configs/global_daily_symbols.json" ]]; then
    echo "Missing global daily symbol config: ${QUANT_WORKFLOW_ROOT}/configs/global_daily_symbols.json" >&2
    exit 2
  fi
  # run_dsa_backtest is intentionally a DSA-native job, not a
  # quant_workflow/scripts wrapper, so validate its entrypoint separately.
  if [[ ! -f "${REPO_ROOT}/main.py" ]]; then
    echo "Missing DSA backtest entrypoint: ${REPO_ROOT}/main.py" >&2
    exit 2
  fi
}

run_py_compile() {
  ensure_python
  section "DSA changed Python compile check"
  cd "${REPO_ROOT}"
  python_files=()
  while IFS= read -r path; do
    if [[ -n "${path}" && -f "${path}" ]]; then
      python_files+=("${path}")
    fi
  done < <(
    {
      git diff --name-only HEAD -- '*.py'
      git ls-files --others --exclude-standard -- '*.py'
    } | sort -u
  )
  if [[ ${#python_files[@]} -eq 0 ]]; then
    echo "No changed Python files to compile."
    return 0
  fi
  "${PYTHON_BIN}" -m py_compile "${python_files[@]}"
}

run_quick_browser_hygiene() {
  section "Data hygiene browser smoke when local DSA is reachable"
  if ! is_local_dsa_reachable; then
    echo "[SKIPPED] Local DSA service is not reachable at ${LOCAL_BASE_URL}; skipping optional quick browser-hygiene smoke." >&2
    echo "Run scripts/verify-local-quant-workbench.sh browser-hygiene or complete when the local service is running." >&2
    return 0
  fi
  ensure_npx
  cd "${WEB_ROOT}"
  npx playwright test e2e/verify-data-hygiene.spec.ts --config playwright.config.ts
}

run_quick() {
  ensure_python
  ensure_npm
  run_py_compile
  check_quant_workflow_scripts
  section "DSA focused backend tests"
  cd "${REPO_ROOT}"
  "${PYTHON_BIN}" -m pytest -q \
    tests/test_api_app_cors.py \
    tests/test_auth_api.py \
    tests/test_build_mlflow_i18n_catalog.py \
    tests/test_mlflow_service.py \
    tests/test_mlflow_ui_proxy.py \
    tests/test_auth.py \
    tests/test_web_bind.py \
    tests/test_main_schedule_mode.py \
    tests/test_alphasift_api.py \
    tests/test_quant_jobs_api.py \
    tests/test_quant_workflow_paths.py \
    tests/test_research_status_api.py \
    tests/test_quant_workflow_docs.py \
    tests/test_safe_urls.py \
    tests/test_stocks_api.py

  section "DSA focused frontend tests"
  cd "${WEB_ROOT}"
  npm test -- --run \
    src/api/__tests__/alphasift.test.ts \
    src/api/__tests__/quant.test.ts \
    src/api/__tests__/research.test.ts \
    src/pages/__tests__/StockScreeningPage.test.tsx \
    src/pages/__tests__/WorkflowPage.test.tsx \
    src/pages/__tests__/ResearchPage.test.tsx \
    src/pages/__tests__/BacktestPage.test.tsx \
    src/pages/__tests__/NotFoundPage.test.tsx \
    src/components/layout/__tests__/SidebarNav.test.tsx \
    src/utils/__tests__/mlflowUiUrl.test.ts

  run_quick_browser_hygiene
}

run_backend() {
  ensure_python
  section "DSA backend ci_gate"
  cd "${REPO_ROOT}"
  PATH="$(dirname "${PYTHON_BIN}"):${PATH}" ./scripts/ci_gate.sh all

  section "DSA full backend pytest"
  cd "${REPO_ROOT}"
  "${PYTHON_BIN}" -m pytest tests -q
}

run_frontend() {
  ensure_npm
  section "DSA frontend vitest"
  cd "${WEB_ROOT}"
  npm test
  section "DSA frontend lint"
  npm run lint
  section "DSA frontend build"
  npm run build
}

run_quant_workflow() {
  ensure_python
  section "quant_workflow script tests"
  if [[ ! -d "${QUANT_WORKFLOW_ROOT}" ]]; then
    echo "quant_workflow root not found: ${QUANT_WORKFLOW_ROOT}" >&2
    exit 2
  fi
  check_quant_workflow_scripts
  cd "${QUANT_WORKFLOW_ROOT}"
  PYTHONPATH="${QUANT_WORKFLOW_ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" -m pytest tests -q
}

run_browser_local() {
  ensure_npx
  ensure_local_dsa
  section "Browser smoke against local DSA"
  cd "${WEB_ROOT}"
  npm run test:smoke:local
  npx playwright test e2e/verify-mlflow-bilingual.spec.ts --config playwright.config.ts
}

run_browser_fake() {
  ensure_npx
  section "Managed fake-analysis browser smoke"
  cd "${WEB_ROOT}"
  local managed_backend_port="${DSA_WEB_MANAGED_SMOKE_BACKEND_PORT:-18011}"
  DSA_WEB_E2E_BASE_URL="http://127.0.0.1:4173" \
  DSA_WEB_SMOKE_BACKEND_PORT="${managed_backend_port}" \
  DSA_WEB_DEV_API_TARGET="http://127.0.0.1:${managed_backend_port}" \
    npm run test:smoke:fake-analysis
}

run_browser_hygiene() {
  ensure_npx
  ensure_local_dsa
  section "Data hygiene browser smoke"
  cd "${WEB_ROOT}"
  npx playwright test e2e/verify-data-hygiene.spec.ts --config playwright.config.ts
}

run_opencode_review() {
  ensure_python
  section "opencode read-only review"
  local bridge="${AGENT_BRIDGE:-${HOME}/.agents/skills/cli-agent-interop/scripts/agent_bridge.py}"
  if [[ ! -f "${bridge}" ]]; then
    echo "agent_bridge.py not found: ${bridge}; opencode-review cannot run." >&2
    echo "WARNING: complete verification gate is NOT satisfied without the working-tree review." >&2
    echo "Install cli-agent-interop or set AGENT_BRIDGE to enable child-agent review before claiming complete verification." >&2
    return 2
  fi
  OPENCODE_REVIEW_TIMEOUT_SECONDS="${OPENCODE_REVIEW_TIMEOUT_SECONDS:-300}" \
    "${PYTHON_BIN}" - "${bridge}" "${REPO_ROOT}" <<'PY'
import os
import subprocess
import sys

bridge = sys.argv[1]
repo_root = sys.argv[2]
try:
    timeout_seconds = int(os.environ.get("OPENCODE_REVIEW_TIMEOUT_SECONDS", "300"))
except ValueError:
    timeout_seconds = 300
timeout_seconds = max(1, timeout_seconds)
prompt = (
    "Review only the current uncommitted working tree for the DSA local quant workbench, "
    "including tracked changes from 'git diff HEAD', the 'git status --short' file list, "
    "and all untracked files reported by that status. In 'git status --short', only lines "
    "beginning with '??' are untracked; lines beginning with 'A ', 'M ', or similar "
    "index/worktree status letters are tracked or staged and must not be reported as untracked. "
    "Do not review historical commit ranges such as HEAD~2..HEAD, and do not report issues "
    "that are absent from the current working tree. Product constraints: this workbench is "
    "intentionally localhost-first; ADMIN_AUTH_ENABLED=true is required for production or "
    "non-local bind hosts; CORS_ALLOW_ALL=true only changes browser CORS headers and must not "
    "bypass the local origin guard; sensitive quant/research/auth/alphasift browser mutations "
    "require a local Origin/Referer and no-origin CLI access is allowed only for non-mutating "
    "health/status-style reads. Treat violations of those constraints as bugs, but do not flag "
    "the constraints themselves as regressions. Focus on bugs, security regressions, missing "
    "tests, browser workflow gaps, newly added but untracked files that must be included, and "
    "whether the verification gates in scripts/verify-local-quant-workbench.sh are adequate. "
    "Do not edit files. Report P0/P1 blockers first with file/line references, then P2/P3 risks."
)
command = [
    sys.executable,
    bridge,
    "run",
    "opencode",
    "--runtime",
    "current",
    "--workdir",
    repo_root,
    "--prompt",
    prompt,
]
try:
    completed = subprocess.run(command, timeout=timeout_seconds, check=False)
except subprocess.TimeoutExpired:
    print(f"opencode-review timed out after {timeout_seconds} seconds.", file=sys.stderr)
    raise SystemExit(124)
raise SystemExit(completed.returncode)
PY
}

run_full() {
  run_backend
  run_frontend
  run_quant_workflow
  run_browser_hygiene
  run_browser_fake
  run_browser_local
}

run_complete() {
  run_full
  run_opencode_review
}

phase="${1:-quick}"
case "${phase}" in
  quick) run_quick ;;
  backend) run_backend ;;
  frontend) run_frontend ;;
  quant-workflow) run_quant_workflow ;;
  browser-local) run_browser_local ;;
  browser-fake) run_browser_fake ;;
  browser-hygiene) run_browser_hygiene ;;
  opencode-review) run_opencode_review ;;
  full) run_full ;;
  complete) run_complete ;;
  -h|--help|help) usage ;;
  *)
    usage >&2
    exit 2
    ;;
esac
