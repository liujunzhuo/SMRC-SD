#!/usr/bin/env bash
# FullPath-SD on ALFWorld with Qwen3-4B: 150 updates, 50-turn horizon, 8 GPUs.
# Default seed is 0. Override ENV_SEED for the other reported seeds.
[[ "${VERBOSE_LAUNCH:-0}" == "1" ]] && set -x
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MAX_STEPS=${MAX_STEPS:-50}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
source "${SCRIPT_DIR}/defaults_qwen3_4b.sh"
export PATH_SDL_MASK_SPECIAL_TOKENS=${PATH_SDL_MASK_SPECIAL_TOKENS:-False}
source "${SCRIPT_DIR}/defaults_alfworld.sh"

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-fullpath_sd_alfworld_qwen3_4b_seed${ENV_SEED}}
export PATH_TEACHER_CONTEXT=gt_path
export PATH_ALIGNMENT_MODE=full_start
export PATH_STATE_SDL_SELECTION=all

mkdir -p "$LOG_DIR"
bash "$SMRC_ALFWORLD_RUNNER" "$@" 2>&1 | tee "${LOG_DIR}/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).log"
