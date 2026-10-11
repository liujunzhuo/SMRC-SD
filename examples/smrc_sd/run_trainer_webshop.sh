#!/usr/bin/env bash
# Keep GPU training in the current environment; simulate WebShop over HTTP.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export WEBSHOP_SERVICE_FILE=${WEBSHOP_SERVICE_FILE:-${REPO_ROOT}/logs/webshop_cpu_services.json}
if [[ ! -f "$WEBSHOP_SERVICE_FILE" ]]; then
    echo "WebShop service registry not found: $WEBSHOP_SERVICE_FILE" >&2
    echo "Start examples/smrc_sd/start_webshop_cpu_service.sh before training." >&2
    exit 1
fi
# Ray workers may use a different working directory.
WEBSHOP_SERVICE_FILE="$(cd "$(dirname "$WEBSHOP_SERVICE_FILE")" && pwd)/$(basename "$WEBSHOP_SERVICE_FILE")"
export WEBSHOP_SERVICE_FILE
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export DATASETS_MAX_WORKERS=${DATASETS_MAX_WORKERS:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
unset RAY_ADDRESS RAY_JOB_CONFIG_JSON_ENV_VAR

exec bash "${SCRIPT_DIR}/run_trainer_alfworld.sh" \
    "++ray_init.runtime_env.env_vars.WEBSHOP_SERVICE_FILE=$WEBSHOP_SERVICE_FILE" \
    "$@"
