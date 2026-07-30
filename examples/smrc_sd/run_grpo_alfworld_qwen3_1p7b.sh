#!/usr/bin/env bash
# GRPO baseline on ALFWorld with Qwen3-1.7B (same stack as paper Path methods).
[[ "${VERBOSE_LAUNCH:-0}" == "1" ]] && set -x
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
export SAVE_FREQ=${SAVE_FREQ:-25}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-250}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-250}

source "${SCRIPT_DIR}/defaults_alfworld.sh"

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-grpo_alfworld_qwen3_1p7b}
export TRAINER_MAIN_MODULE=verl.trainer.main_path_opd
export PATH_OPD_ENABLE=False
export PATH_TEACHER_CONTEXT=none
export PATH_OPD_COEF=0.0
export PATH_STRICT_GT=False
export PATH_ENABLE_WALKTHROUGH_TRACE_RECOVERY=False
export TRAJECTORY_INDEX_PATH=null
export INCLUDE_SDAR_CONFIG=False

mkdir -p "$LOG_DIR"
bash "$SMRC_ALFWORLD_RUNNER" "$@" 2>&1 | tee "${LOG_DIR}/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).log"
