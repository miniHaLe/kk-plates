#!/bin/sh
# Portable supervisor to keep the dashboard running.
# Restarts the app whenever it exits, with a small backoff.

# Safe options where available
# -e: exit on error; -u: undefined as error (not all /bin/sh support -u reliably)
set -e

ROOT_DIR="/home/hale/hale"
PYTHON_BIN="${PYTHON:-python3}"
CMD="KICHI_NONINTERACTIVE=1 KICHI_SUPERVISED=1 ${PYTHON_BIN} ${ROOT_DIR}/run.py --mode dashboard"

LOG_DIR="${ROOT_DIR}/logs"
OUT_LOG="${LOG_DIR}/dashboard_stdout.log"
ERR_LOG="${LOG_DIR}/dashboard_stderr.log"

mkdir -p "${LOG_DIR}"

backoff=2
max_backoff=60

trap 'echo "[supervisor] Caught signal, stopping." | tee -a "${OUT_LOG}"; exit 0' INT TERM

echo "[supervisor] Starting dashboard supervisor at $(date -Iseconds)" | tee -a "${OUT_LOG}"

while true; do
  start_ts=$(date +%s)
  echo "[supervisor] Launching: ${CMD} at $(date -Iseconds)" | tee -a "${OUT_LOG}"
  # Run the app, redirecting stdout/stderr to logs
  # shellcheck disable=SC2086
  sh -c "${CMD}" >>"${OUT_LOG}" 2>>"${ERR_LOG}" || true
  exit_code=$?
  end_ts=$(date +%s)
  runtime=$(( end_ts - start_ts ))
  echo "[supervisor] Process exited with code ${exit_code} after ${runtime}s at $(date -Iseconds)" | tee -a "${OUT_LOG}"

  # Exponential backoff if it fails immediately
  if [ "${runtime}" -lt 10 ]; then
    backoff=$(( backoff * 2 ))
    if [ "${backoff}" -gt "${max_backoff}" ]; then backoff=${max_backoff}; fi
  else
    backoff=2
  fi

  echo "[supervisor] Restarting in ${backoff}s..." | tee -a "${OUT_LOG}"
  sleep ${backoff}
done


