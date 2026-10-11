#!/usr/bin/env bash
# Qwen3-4B FTRL on 8 x A100/H100 80GB (CUDA).
# Callers set FTRL_VARIANT to fullpath or exact.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"
: "${FTRL_VARIANT:?Set FTRL_VARIANT to fullpath or exact}"
: "${FTRL_ROOT:?Set FTRL_ROOT to the prepared FTRL asset directory}"
case "$FTRL_VARIANT" in
    fullpath) selection=all ;;
    exact) selection=match_only ;;
    *) echo "FTRL_VARIANT must be fullpath or exact" >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export ENV_SEED=${ENV_SEED:-0}
export PYTHONHASHSEED="$ENV_SEED"
references=${FTRL_REFERENCES:-${FTRL_ROOT}/normalized/references_normalized_v1.jsonl}
state_defaults=${FTRL_STATE_DEFAULTS:-${FTRL_ROOT}/normalized/state_defaults_v1.json}
train_files=${FTRL_TRAIN_FILES:-${FTRL_ROOT}/train_rl.parquet}
val_files=${FTRL_VAL_FILES:-${FTRL_ROOT}/test_rl.parquet}
experiment_name=${EXPERIMENT_NAME:-${FTRL_VARIANT}_sd_ftrl_qwen3_4b_seed${ENV_SEED}}
suite_dir=${PATH_OPD_SUITE_DIR:-logs/smrc_sd_ftrl}
mkdir -p "$suite_dir"

exec python3 -m verl.trainer.main_ftrl_smrc \
    "env.ftrl.variant=${FTRL_VARIANT}" \
    "env.ftrl.url=${FTRL_SERVICE_URL:-http://127.0.0.1:28980}" \
    "env.ftrl.references=${references}" \
    "env.ftrl.state_defaults=${state_defaults}" \
    "env.seed=${ENV_SEED}" \
    "data.seed=${DATA_SEED:-$ENV_SEED}" \
    "actor_rollout_ref.rollout.seed=${ROLLOUT_SEED:-$ENV_SEED}" \
    "actor_rollout_ref.rollout.generation_seed=${GENERATION_SEED:-$ENV_SEED}" \
    "data.train_files=${train_files}" \
    "data.val_files=${val_files}" \
    "algorithm.path_opd.state_sdl_selection=${selection}" \
    "actor_rollout_ref.model.path=${MODEL_PATH:-Qwen/Qwen3-4B}" \
    "trainer.n_gpus_per_node=${N_GPUS_PER_NODE:-8}" \
    "trainer.experiment_name=${experiment_name}" \
    "trainer.default_local_dir=${CHECKPOINT_DIR:-${suite_dir}/${experiment_name}/checkpoints}" \
    "trainer.rollout_data_dir=${suite_dir}/${experiment_name}/rollouts" \
    "trainer.validation_data_dir=${suite_dir}/${experiment_name}/validation" \
    "$@"
