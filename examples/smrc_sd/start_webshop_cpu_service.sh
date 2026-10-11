#!/usr/bin/env bash
# Start isolated CPU WebShop simulators for the training process.
# The GPU environment does not need to import the WebShop stack.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
: "${WEBSHOP_CPU_PYTHON:?Set WEBSHOP_CPU_PYTHON to the Python 3.10 simulator interpreter}"
: "${WEBSHOP_DATA_DIR:?Set WEBSHOP_DATA_DIR to the WebShop-small data directory}"

count=${WEBSHOP_SERVICE_COUNT:-8}
base_port=${WEBSHOP_BASE_PORT:-28180}
registry=${WEBSHOP_SERVICE_FILE:-${REPO_ROOT}/logs/webshop_cpu_services.json}
log_dir=${WEBSHOP_SERVICE_LOG_DIR:-${REPO_ROOT}/logs/webshop_cpu}
prefix="$(cd "$(dirname "${WEBSHOP_CPU_PYTHON}")/.." && pwd)"
mkdir -p "$(dirname "$registry")" "$log_dir"

urls=()
pids=()
for ((i = 0; i < count; i++)); do
    port=$((base_port + i))
    log="${log_dir}/server_${port}.log"
    env \
        WEBSHOP_DATA_DIR="$WEBSHOP_DATA_DIR" \
        PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}" \
        PYTHONUNBUFFERED=1 \
        CUDA_VISIBLE_DEVICES="" \
        JAVA_HOME="${JAVA_HOME:-$prefix}" \
        PATH="${prefix}/bin:${PATH}" \
        OMP_NUM_THREADS=1 \
        OPENBLAS_NUM_THREADS=1 \
        MKL_NUM_THREADS=1 \
        TOKENIZERS_PARALLELISM=false \
        _JAVA_OPTIONS="${_JAVA_OPTIONS:--XX:+UseSerialGC -Xss512k -Xmx1g}" \
        "${WEBSHOP_CPU_PYTHON}" -m scripts.webshop_server --port "$port" >>"$log" 2>&1 &
    pids+=("$!")
    urls+=("http://127.0.0.1:${port}")
done

python3 - "$registry" "${urls[@]}" <<'PY'
import json, sys
path, *urls = sys.argv[1:]
open(path, "w").write(json.dumps({"urls": urls}, indent=2) + "\n")
PY

deadline=$((SECONDS + 180))
pending=("${urls[@]}")
while (( SECONDS < deadline && ${#pending[@]} )); do
    next=()
    for url in "${pending[@]}"; do
        if ! curl -fsS --noproxy '*' "${url}/health" >/dev/null 2>&1; then
            next+=("$url")
        fi
    done
    pending=("${next[@]}")
    if (( ${#pending[@]} )); then
        sleep 1
    fi
done
if (( ${#pending[@]} )); then
    echo "WebShop CPU services did not become healthy: ${pending[*]}" >&2
    echo "Logs: ${log_dir}" >&2
    exit 1
fi
echo "WEBSHOP_SERVICE_FILE=${registry}"
echo "Started ${count} WebShop CPU services, pids: ${pids[*]}"
