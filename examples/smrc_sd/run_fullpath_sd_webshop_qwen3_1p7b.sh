#!/usr/bin/env bash
# FullPath-SD on WebShop with Qwen3-1.7B (paper main-table: update 150).
[[ "${VERBOSE_LAUNCH:-0}" == "1" ]] && set -x
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-150}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-${TOTAL_TRAINING_STEPS}}
SAVE_FREQ=${SAVE_FREQ:-25}

source "${SCRIPT_DIR}/defaults_webshop.sh"

export TOTAL_TRAINING_STEPS
export TOTAL_EPOCHS
export SAVE_FREQ
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-fullpath_sd_webshop_qwen3_1p7b}
export PATH_TEACHER_CONTEXT=gt_path
export PATH_ALIGNMENT_MODE=full_start
export PATH_ENABLE_WALKTHROUGH_TRACE_RECOVERY=False
export PATH_RECOVERY_MATCH_MODE=history
export PATH_STATE_SDL_SELECTION=all

mkdir -p "$LOG_DIR"
bash "$SMRC_WEBSHOP_RUNNER" "$@" 2>&1 | tee "${LOG_DIR}/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).log"
