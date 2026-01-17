#!/bin/bash
set -euo pipefail

# Submit validate_range jobs with walltime=24:00:00.
# Defaults: TOTAL=1300000, CHUNK=100000, MAX_INFLIGHT=8.
# Optional env overrides:
#   START_AT, END_AT, TOTAL, CHUNK, MAX_INFLIGHT, WALLTIME, PBS_SCRIPT
#   DATASET_REPO, OUTPUT_DIR, OUTPUT_PARQUET, LOG_DIR, LOG_EVERY,
#   TIMEOUT_SECONDS, MAX_OUTPUT_CHARS, CODE_MAX_CHARS,
#   NEMO_SKILLS_SANDBOX_HOST, NEMO_SKILLS_SANDBOX_PORT, VENV_DIR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PBS_SCRIPT="${PBS_SCRIPT:-${SCRIPT_DIR}/run_validate_range.pbs}"

TOTAL="${TOTAL:-1300000}"
START_AT="${START_AT:-0}"
END_AT="${END_AT:-$TOTAL}"
CHUNK="${CHUNK:-100000}"
MAX_INFLIGHT="${MAX_INFLIGHT:-8}"
WALLTIME="${WALLTIME:-24:00:00}"

if [[ "${START_AT}" -ge "${END_AT}" ]]; then
  echo "START_AT must be < END_AT (START_AT=${START_AT}, END_AT=${END_AT})" >&2
  exit 1
fi

qstat_inflight() {
  qstat -u "$USER" | awk 'NR>5 && $3=="rt_HC" {c++} END{print c+0}'
}

for ((start=START_AT; start<END_AT; start+=CHUNK)); do
  end=$((start+CHUNK))
  if [[ "$end" -gt "$END_AT" ]]; then end="$END_AT"; fi

  while [[ "$(qstat_inflight)" -ge "$MAX_INFLIGHT" ]]; do
    sleep 60
  done

  vars=("START_ID=${start}" "END_ID=${end}")
  for name in DATASET_REPO OUTPUT_DIR OUTPUT_PARQUET LOG_DIR LOG_EVERY \
    TIMEOUT_SECONDS MAX_OUTPUT_CHARS CODE_MAX_CHARS \
    NEMO_SKILLS_SANDBOX_HOST NEMO_SKILLS_SANDBOX_PORT VENV_DIR; do
    if [[ -n "${!name:-}" ]]; then
      vars+=("${name}=${!name}")
    fi
  done

  qsub -l walltime="${WALLTIME}" -v "$(IFS=,; echo "${vars[*]}")" "${PBS_SCRIPT}"
done
