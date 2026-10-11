#!/usr/bin/env bash
# Exact-state SMRC-SD on FTRL with Qwen3-4B. Default seed is 0.
# The reported main-table run used seed 42; set ENV_SEED=42 to match that run.
export FTRL_VARIANT=exact
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_ftrl_qwen3_4b.sh" "$@"
