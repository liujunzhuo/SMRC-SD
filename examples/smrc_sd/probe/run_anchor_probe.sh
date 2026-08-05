#!/usr/bin/env bash
# Fixed-anchor teacher intervention probe used in the paper (1,200 ALFWorld turns).
# Requires GRPO rollouts at updates 5/100/150/200/250 and a GRPO update-250 actor checkpoint.
set -euo pipefail
[[ "${VERBOSE_LAUNCH:-0}" == "1" ]] && set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export ALFWORLD_DATA=${ALFWORLD_DATA:-${REPO_ROOT}/data/raw/alfworld}
export PROBE_ROOT=${PROBE_ROOT:-logs/smrc_anchor_probe}
export MANIFEST=${MANIFEST:-${PROBE_ROOT}/anchors/grpo_5_100_150_200_250_len10_seed0.jsonl}
export TRAJECTORY_INDEX=${TRAJECTORY_INDEX:-examples/smrc_sd/data/alfworld_verified_walkthrough_prefixes.jsonl}
export GRPO_ROLLOUT_DIR=${GRPO_ROLLOUT_DIR:-logs/smrc_sd_alfworld/rollouts/grpo_alfworld_qwen3_1p7b}
export BASE_MODEL=${BASE_MODEL:-${MODEL_PATH:-Qwen/Qwen3-1.7B}}
export GRPO250_ACTOR=${GRPO250_ACTOR:-checkpoints/smrc_sd_alfworld/grpo_alfworld_qwen3_1p7b/global_step_250/actor}
export GRPO250_HF=${GRPO250_HF:-${PROBE_ROOT}/merged_models/grpo_global_step_250_actor_hf}
export BATCH_SIZE=${BATCH_SIZE:-4}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
export NPROC=${NPROC:-4}

mkdir -p "$(dirname "$MANIFEST")"
python3 "${SCRIPT_DIR}/build_anchor_manifest.py" \
    --rollout-jsonl "${GRPO_ROLLOUT_DIR}/5.jsonl" "${GRPO_ROLLOUT_DIR}/100.jsonl" "${GRPO_ROLLOUT_DIR}/150.jsonl" "${GRPO_ROLLOUT_DIR}/200.jsonl" "${GRPO_ROLLOUT_DIR}/250.jsonl" \
    --trajectory-index "$TRAJECTORY_INDEX" \
    --output "$MANIFEST" \
    --max-observed-path-actions 10

run_score() {
    local label="$1"
    local model_path="$2"
    local output_dir="${PROBE_ROOT}/scores/${label}"
    torchrun --standalone --nproc_per_node="${NPROC}" "${SCRIPT_DIR}/score_anchor_suite.py" \
        --anchor-manifest "$MANIFEST" \
        --model-path "$model_path" \
        --backend fsdp \
        --batch-size "$BATCH_SIZE" \
        --max-model-len "$MAX_MODEL_LEN" \
        --output-dir "$output_dir"
}

run_score baseline "$BASE_MODEL"

if [[ ! -f "${GRPO250_HF}/config.json" ]]; then
    test -d "$GRPO250_ACTOR"
    PYTHONPATH=. python3 scripts/model_merger.py merge --backend fsdp --local_dir "$GRPO250_ACTOR" --target_dir "$GRPO250_HF"
fi
run_score grpo250 "$GRPO250_HF"
