#!/usr/bin/env bash
# FullPath-SD on ALFWorld with Qwen3-1.7B (paper main-table: update 250).
[[ "${VERBOSE_LAUNCH:-0}" == "1" ]] && set -x
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
export SAVE_FREQ=${SAVE_FREQ:-25}

source "${SCRIPT_DIR}/defaults_alfworld.sh"

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-fullpath_sd_alfworld_qwen3_1p7b}
export PATH_TEACHER_CONTEXT=gt_path
export PATH_ALIGNMENT_MODE=full_start
export PATH_STATE_SDL_SELECTION=all

mkdir -p "$LOG_DIR"
bash "$SMRC_ALFWORLD_RUNNER" "$@" 2>&1 | tee "${LOG_DIR}/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).log"
