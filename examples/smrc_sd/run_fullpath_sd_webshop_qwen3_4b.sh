#!/usr/bin/env bash
# FullPath-SD on WebShop with Qwen3-4B: 150 updates, 15-turn horizon, 8 GPUs.
# The simulator runs in a separate CPU service. Default seed is 0.
[[ "${VERBOSE_LAUNCH:-0}" == "1" ]] && set -x
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/defaults_qwen3_4b.sh"
source "${SCRIPT_DIR}/defaults_webshop.sh"

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-fullpath_sd_webshop_qwen3_4b_seed${ENV_SEED}}
export PATH_TEACHER_CONTEXT=gt_path
export PATH_ALIGNMENT_MODE=full_start
export PATH_ENABLE_WALKTHROUGH_TRACE_RECOVERY=False
export PATH_RECOVERY_MATCH_MODE=history
export PATH_STATE_SDL_SELECTION=all
export PATH_RECOVERY_STATE_DOMAIN=webshop

mkdir -p "$LOG_DIR"
bash "$SMRC_WEBSHOP_RUNNER" "$@" 2>&1 | tee "${LOG_DIR}/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).log"
