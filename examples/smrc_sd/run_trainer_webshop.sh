#!/usr/bin/env bash

# WebShop runner wrapper. Prefer an existing WebShop conda env; fall back to
# the current Python if CONDA_BIN / CONDA_ENV are unset or unavailable.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
GENERIC_RUNNER="${SCRIPT_DIR}/run_trainer_alfworld.sh"
CONDA_BIN=${CONDA_BIN:-}
CONDA_ENV=${CONDA_ENV:-verl-agent-webshop}

if [[ -n "${CONDA_BIN}" && -x "${CONDA_BIN}" && "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
    unset RAY_ADDRESS
    unset RAY_JOB_CONFIG_JSON_ENV_VAR
    exec "$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" bash "$0" "$@"
fi

if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export JAVA_HOME=${JAVA_HOME:-$CONDA_PREFIX}
    export PATH="$JAVA_HOME/bin:$PATH"
fi
export _JAVA_OPTIONS="${_JAVA_OPTIONS:--XX:+UseSerialGC -Xss512k -Xmx1g}"
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export DATASETS_MAX_WORKERS=${DATASETS_MAX_WORKERS:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
unset RAY_ADDRESS
unset RAY_JOB_CONFIG_JSON_ENV_VAR

exec bash "$GENERIC_RUNNER" "$@"
