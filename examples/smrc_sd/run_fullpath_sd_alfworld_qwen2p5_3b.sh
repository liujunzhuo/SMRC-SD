#!/usr/bin/env bash
# FullPath-SD on ALFWorld with Qwen2.5-3B-Instruct (paper main-table: update 150).
[[ "${VERBOSE_LAUNCH:-0}" == "1" ]] && set -x
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
export SAVE_FREQ=${SAVE_FREQ:-25}
export TEST_FREQ=${TEST_FREQ:-25}

export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-150}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-150}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.50}
export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-20000}
export MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-24576}
export PATH_OPD_DEFAULT_SUITE_DIR=${PATH_OPD_DEFAULT_SUITE_DIR:-logs/smrc_sd_alfworld_qwen2p5_3b}
export ALFWORLD_REQUIRE_THINK_TAGS=${ALFWORLD_REQUIRE_THINK_TAGS:-True}

source "${SCRIPT_DIR}/defaults_alfworld.sh"

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-fullpath_sd_alfworld_qwen2p5_3b}
export PATH_TEACHER_CONTEXT=gt_path
export PATH_ALIGNMENT_MODE=full_start
export PATH_STATE_SDL_SELECTION=all

mkdir -p "$LOG_DIR"
bash "$SMRC_ALFWORLD_RUNNER" "$@" 2>&1 | tee "${LOG_DIR}/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).log"
