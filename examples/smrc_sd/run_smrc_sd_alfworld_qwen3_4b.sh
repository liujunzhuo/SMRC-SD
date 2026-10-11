#!/usr/bin/env bash
# SMRC-SD on ALFWorld with Qwen3-4B: 150 updates, 50-turn horizon, 8 GPUs.
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

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-smrc_sd_alfworld_qwen3_4b_seed${ENV_SEED}}
export PATH_TEACHER_CONTEXT=gt_recovery_aligned
export PATH_ALIGNMENT_MODE=dynamic
export PATH_ENABLE_WALKTHROUGH_TRACE_RECOVERY=True
export PATH_RECOVERY_MATCH_MODE=state
export PATH_ENABLE_HISTORY_SUFFIX_ALIGNMENT=False
export PATH_RECOVERY_USE_LAST_PROGRESS_OBSERVATION_ON_NO_PROGRESS=False
export PATH_RECOVERY_ALFWORLD_OBJECT_MATCH_MODE=exact
export PATH_STATE_SDL_SELECTION=match_only
export PATH_STATE_MATCH_CONTEXT=recovery
export PATH_RECOVERY_PROMPT_STYLE=state_summary_candidate
export PATH_RECOVERY_GUIDANCE_STYLE=path_on_or_off
export PATH_SDL_SAMPLE_FILTER=all
export PATH_SDL_LOSS_NORMALIZATION=response_token_mean
export PATH_SDL_CONFIDENCE_WEIGHTING=False
export PATH_SDL_DROP_REPEATED_ACTION_AFTER_TURN=False
export PATH_POLICY_LOSS_FILTER=none
export PATH_RECOVERY_CORRECTION_ONLY=False
export PATH_RECOVERY_SUCCESS_CONFIRMATION_SDL_WEIGHT=1.0

mkdir -p "$LOG_DIR"
bash "$SMRC_ALFWORLD_RUNNER" "$@" 2>&1 | tee "${LOG_DIR}/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).log"
