#!/usr/bin/env bash
# Install (or preview) the launchd job that runs the daily A-share data update.
#
# The daily pipeline refreshes independent source caches, builds merged and
# cleaned candidates, validates benchmark/quality/Qlib state, and publishes
# official providers only after every required check passes. It records a
# controlled skip instead of touching shared data while robust research runs.
#
# Baostock uses anonymous login, so this job needs no credentials. The plist is
# machine-specific (absolute paths), so it is generated here from the tracked
# template rather than committed with resolved paths.
#
# Usage:
#   scripts/launchd/install-quant-daily-update.sh            # write plist, print load command
#   scripts/launchd/install-quant-daily-update.sh --load     # write plist and bootstrap it now
#   scripts/launchd/install-quant-daily-update.sh --dry-run  # print resolved plist, write nothing
#   scripts/launchd/install-quant-daily-update.sh --uninstall # bootout and remove the plist

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TEMPLATE="${SCRIPT_DIR}/com.dsa.quant-daily-update.plist.template"
LABEL="com.dsa.quant-daily-update"
PLIST_DEST="${HOME}/Library/LaunchAgents/${LABEL}.plist"

PYTHON_BIN="${DSA_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
UPDATE_SCRIPT="${REPO_ROOT}/quant_workflow/scripts/run_daily_update.py"
WORKDIR="${REPO_ROOT}/quant_workflow"
LOG_DIR="${WORKDIR}/logs"
STDOUT_LOG="${LOG_DIR}/quant_daily_update.stdout.log"
STDERR_LOG="${LOG_DIR}/quant_daily_update.stderr.log"

MODE="install"
for arg in "$@"; do
  case "${arg}" in
    --load) MODE="load" ;;
    --dry-run) MODE="dry-run" ;;
    --uninstall) MODE="uninstall" ;;
    *) echo "Unknown option: ${arg}" >&2; exit 2 ;;
  esac
done
uid="$(id -u)"

if [[ "${MODE}" == "uninstall" ]]; then
  launchctl bootout "gui/${uid}/${LABEL}" 2>/dev/null || true
  if [[ -f "${PLIST_DEST}" ]]; then
    rm -f "${PLIST_DEST}"
    echo "Removed ${PLIST_DEST}"
  else
    echo "No plist at ${PLIST_DEST}; nothing to remove."
  fi
  exit 0
fi

if [[ ! -f "${TEMPLATE}" ]]; then
  echo "Template not found: ${TEMPLATE}" >&2
  exit 1
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not found or not executable: ${PYTHON_BIN}" >&2
  echo "Set DSA_PYTHON to override." >&2
  exit 1
fi
if [[ ! -f "${UPDATE_SCRIPT}" ]]; then
  echo "Update script not found: ${UPDATE_SCRIPT}" >&2
  exit 1
fi

rendered="$(sed \
  -e "s#__HOME__#${HOME}#g" \
  -e "s#__PYTHON__#${PYTHON_BIN}#g" \
  -e "s#__SCRIPT__#${UPDATE_SCRIPT}#g" \
  -e "s#__WORKDIR__#${WORKDIR}#g" \
  -e "s#__STDOUT__#${STDOUT_LOG}#g" \
  -e "s#__STDERR__#${STDERR_LOG}#g" \
  "${TEMPLATE}")"

if [[ "${MODE}" == "dry-run" ]]; then
  echo "# Resolved plist (not written):"
  echo "${rendered}"
  exit 0
fi

mkdir -p "${LOG_DIR}"
mkdir -p "$(dirname "${PLIST_DEST}")"
printf '%s\n' "${rendered}" > "${PLIST_DEST}"
echo "Wrote ${PLIST_DEST}"

if [[ "${MODE}" == "load" ]]; then
  launchctl bootout "gui/${uid}/${LABEL}" 2>/dev/null || true
  launchctl bootstrap "gui/${uid}" "${PLIST_DEST}"
  echo "Loaded ${LABEL}. Schedule: weekdays 18:40 local."
  echo "Run once now:  launchctl kickstart gui/${uid}/${LABEL}"
else
  echo "To load it:    launchctl bootstrap gui/${uid} ${PLIST_DEST}"
  echo "Run once now:  launchctl kickstart gui/${uid}/${LABEL}"
  echo "To remove:     scripts/launchd/install-quant-daily-update.sh --uninstall"
fi
