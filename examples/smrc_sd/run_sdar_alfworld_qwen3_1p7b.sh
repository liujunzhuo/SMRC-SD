#!/usr/bin/env bash
# SDAR reproduction on ALFWorld with Qwen3-1.7B (skill teacher + SDAR token gate).
[[ "${VERBOSE_LAUNCH:-0}" == "1" ]] && set -x
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
export SAVE_FREQ=${SAVE_FREQ:-25}

source "${SCRIPT_DIR}/defaults_alfworld.sh"

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-sdar_alfworld_qwen3_1p7b}
export PATH_TEACHER_CONTEXT=skill
export PATH_ALIGNMENT_MODE=full_start
export PATH_LOSS_TYPE=sdar
export PATH_OPD_COEF=${PATH_OPD_COEF:-0.01}
export PATH_GATE_BETA=${PATH_GATE_BETA:-5.0}
export INCLUDE_SDAR_CONFIG=True
export TRAJECTORY_INDEX_PATH=null

mkdir -p "$LOG_DIR"
bash "$SMRC_ALFWORLD_RUNNER" "$@" 2>&1 | tee "${LOG_DIR}/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).log"
