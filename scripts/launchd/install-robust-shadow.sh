#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LABEL="com.dsa.robust-shadow"
TEMPLATE="${SCRIPT_DIR}/${LABEL}.plist.template"
DEST="${HOME}/Library/LaunchAgents/${LABEL}.plist"
PYTHON_BIN="${DSA_ROBUST_PYTHON:-${REPO_ROOT}/.venv-mps-dnn/bin/python}"
SCRIPT="${REPO_ROOT}/quant_workflow/scripts/run_robust_shadow_schedule.py"
CONFIG="${DSA_ROBUST_CONFIG:-${REPO_ROOT}/quant_workflow/configs/robust_strategy_mps_dnn.yaml}"
WORKDIR="${REPO_ROOT}/quant_workflow"
LOG_DIR="${WORKDIR}/logs"
MODE="install"
for arg in "$@"; do case "${arg}" in --load) MODE=load ;; --dry-run) MODE=dry-run ;; --uninstall) MODE=uninstall ;; *) echo "Unknown option: ${arg}" >&2; exit 2 ;; esac; done
uid="$(id -u)"
if [[ "${MODE}" == uninstall ]]; then launchctl bootout "gui/${uid}/${LABEL}" 2>/dev/null || true; rm -f "${DEST}"; exit 0; fi
for path in "${PYTHON_BIN}" "${SCRIPT}" "${CONFIG}" "${TEMPLATE}"; do [[ -e "${path}" ]] || { echo "Required path missing: ${path}" >&2; exit 1; }; done
rendered="$(sed -e "s#__HOME__#${HOME}#g" -e "s#__PYTHON__#${PYTHON_BIN}#g" -e "s#__SCRIPT__#${SCRIPT}#g" -e "s#__CONFIG__#${CONFIG}#g" -e "s#__WORKDIR__#${WORKDIR}#g" -e "s#__STDOUT__#${LOG_DIR}/robust_shadow.stdout.log#g" -e "s#__STDERR__#${LOG_DIR}/robust_shadow.stderr.log#g" "${TEMPLATE}")"
if [[ "${MODE}" == dry-run ]]; then printf '%s\n' "${rendered}"; exit 0; fi
mkdir -p "${LOG_DIR}" "$(dirname "${DEST}")"; printf '%s\n' "${rendered}" > "${DEST}"
if [[ "${MODE}" == load ]]; then launchctl bootout "gui/${uid}/${LABEL}" 2>/dev/null || true; launchctl bootstrap "gui/${uid}" "${DEST}"; fi
echo "Installed ${LABEL}; weekdays 19:10 local, after the 18:40 data update."
