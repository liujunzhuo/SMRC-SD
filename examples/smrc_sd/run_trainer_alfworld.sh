#!/usr/bin/env bash
if [[ "${VERBOSE_LAUNCH:-0}" == "1" ]]; then
    set -x
fi
set -e
if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
    echo "Unsetting PYTORCH_CUDA_ALLOC_CONF because expandable_segments is incompatible with vLLM CuMemAllocator."
    unset PYTORCH_CUDA_ALLOC_CONF
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

ENGINE=vllm
if [[ $# -gt 0 && "$1" != *=* && "$1" != +*=* ]]; then
    ENGINE=$1
    shift
fi
trainer_main_module=${TRAINER_MAIN_MODULE:-verl.trainer.main_path_opd}
case "$trainer_main_module" in
    verl.trainer.main_path_opd|verl.trainer.main_ppo) ;;
    *)
        echo "Unsupported TRAINER_MAIN_MODULE: $trainer_main_module" >&2
        exit 2
        ;;
esac

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}
export ALFWORLD_DATA=${ALFWORLD_DATA:-${REPO_ROOT}/data/raw/alfworld}
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}

num_cpus_per_env_worker=${NUM_CPUS_PER_ENV_WORKER:-0.1}
model_path=${MODEL_PATH:-Qwen/Qwen3-1.7B}
train_data_size=${TRAIN_DATA_SIZE:-16}
val_data_size=${VAL_DATA_SIZE:-128}
group_size=${GROUP_SIZE:-8}
total_training_steps=${TOTAL_TRAINING_STEPS:-250}
total_epochs=${TOTAL_EPOCHS:-250}
max_steps=${MAX_STEPS:-30}
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-512}
env_history_length=${ENV_HISTORY_LENGTH:-2}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-256}
ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-8}
loss_agg_mode=${LOSS_AGG_MODE:-token-mean}
actor_use_kl_loss=${ACTOR_USE_KL_LOSS:-False}
actor_kl_loss_coef=${ACTOR_KL_LOSS_COEF:-0.0}
actor_kl_loss_type=${ACTOR_KL_LOSS_TYPE:-low_var_kl}
log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-64}
actor_strategy=${ACTOR_STRATEGY:-fsdp}
ref_strategy=${REF_STRATEGY:-fsdp}
actor_param_offload=${ACTOR_PARAM_OFFLOAD:-False}
actor_optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD:-False}
ref_param_offload=${REF_PARAM_OFFLOAD:-True}
use_dynamic_bsz=${USE_DYNAMIC_BSZ:-True}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-28000}
log_prob_max_token_len_per_gpu=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-65536}
enable_activation_offload=${ENABLE_ACTIVATION_OFFLOAD:-False}
rollout_free_cache_engine=${ROLLOUT_FREE_CACHE_ENGINE:-False}
rollout_keep_engine_awake_during_multiturn=${ROLLOUT_KEEP_ENGINE_AWAKE_DURING_MULTITURN:-False}
rollout_enable_chunked_prefill=${ROLLOUT_ENABLE_CHUNKED_PREFILL:-False}
rollout_gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}
max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-32768}
rollout_temperature=${ROLLOUT_TEMPERATURE:-1.0}
rollout_top_p=${ROLLOUT_TOP_P:-1.0}
rollout_top_k=${ROLLOUT_TOP_K:--1}
val_temperature=${VAL_TEMPERATURE:-0.4}
val_do_sample=${VAL_DO_SAMPLE:-True}
test_freq=${TEST_FREQ:-25}
log_val_generations=${LOG_VAL_GENERATIONS:-0}
n_gpus_per_node=${N_GPUS_PER_NODE:-2}
env_seed=${ENV_SEED:-0}
logger=${LOGGER:-"['console']"}
project_name=${PROJECT_NAME:-smrc_sd_alfworld}
experiment_name=${EXPERIMENT_NAME:-smrc_sd}
path_opd_default_suite_dir=${PATH_OPD_DEFAULT_SUITE_DIR:-logs/smrc_sd_alfworld}
suite_dir=${PATH_OPD_SUITE_DIR:-${path_opd_default_suite_dir}}
rollout_data_dir=${ROLLOUT_DATA_DIR:-${suite_dir}/rollouts/${experiment_name}}
validation_data_dir=${VALIDATION_DATA_DIR:-${suite_dir}/validation/${experiment_name}}
rollout_dump_freq=${ROLLOUT_DUMP_FREQ:-5}
rollout_only=${ROLLOUT_ONLY:-False}
val_before_train=${VAL_BEFORE_TRAIN:-True}
save_freq=${SAVE_FREQ:-200}
max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP:-2}
max_critic_ckpt_to_keep=${MAX_CRITIC_CKPT_TO_KEEP:-2}
alfworld_prompt_style=${ALFWORLD_PROMPT_STYLE:-concise_thought}
# Empty => leave env.*.require_think_tags unset (null), which follows
# data.apply_chat_template_kwargs.enable_thinking. Set True/False explicitly
# for Qwen2.5 think-tag prompts while keeping enable_thinking=False.
alfworld_require_think_tags=${ALFWORLD_REQUIRE_THINK_TAGS:-}
webshop_require_think_tags=${WEBSHOP_REQUIRE_THINK_TAGS:-}
alfworld_train_game_files=${ALFWORLD_TRAIN_GAME_FILES:-null}
penalize_extra_output_after_action=${PENALIZE_EXTRA_OUTPUT_AFTER_ACTION:-False}
task_env_name=${TASK_ENV_NAME:-alfworld/AlfredTWEnv}
task_skills_dir=${TASK_SKILLS_DIR:-skills/alfworld}
webshop_use_small=${WEBSHOP_USE_SMALL:-True}
webshop_human_goals=${WEBSHOP_HUMAN_GOALS:-False}
webshop_oracle_query_variant=${WEBSHOP_ORACLE_QUERY_VARIANT:-full_name}
webshop_goal_order_seed=${WEBSHOP_GOAL_ORDER_SEED:-0}
webshop_validation_goal_seed=${WEBSHOP_VALIDATION_GOAL_SEED:-1000}
webshop_oracle_manifest_path=${WEBSHOP_ORACLE_MANIFEST_PATH:-null}
webshop_validation_manifest_path=${WEBSHOP_VALIDATION_MANIFEST_PATH:-null}

path_teacher_context=${PATH_TEACHER_CONTEXT:-gt_path}
path_loss_type=${PATH_LOSS_TYPE:-sdar}
path_opd_coef=${PATH_OPD_COEF:-0.01}
path_opd_enable=${PATH_OPD_ENABLE:-True}
path_sdl_sample_filter=${PATH_SDL_SAMPLE_FILTER:-all}
path_sdl_token_scope=${PATH_SDL_TOKEN_SCOPE:-all}
path_sdl_mask_special_tokens=${PATH_SDL_MASK_SPECIAL_TOKENS:-False}
path_sdl_loss_normalization=${PATH_SDL_LOSS_NORMALIZATION:-selected_token_mean}
path_sdl_loss_mode=${PATH_SDL_LOSS_MODE:-chosen_token_k3}
path_state_sdl_selection=${PATH_STATE_SDL_SELECTION:-all}
path_state_match_context=${PATH_STATE_MATCH_CONTEXT:-recovery}
path_recovery_prompt_style=${PATH_RECOVERY_PROMPT_STYLE:-current}
path_recovery_guidance_style=${PATH_RECOVERY_GUIDANCE_STYLE:-style_default}
path_enable_walkthrough_trace_recovery=${PATH_ENABLE_WALKTHROUGH_TRACE_RECOVERY:-True}
path_recovery_match_mode=${PATH_RECOVERY_MATCH_MODE:-state}
path_recovery_state_domain=${PATH_RECOVERY_STATE_DOMAIN:-alfworld}
path_recovery_alfworld_object_match_mode=${PATH_RECOVERY_ALFWORLD_OBJECT_MATCH_MODE:-exact}
path_recovery_webshop_match_mode=${PATH_RECOVERY_WEBSHOP_MATCH_MODE:-goal_progress}
path_gate_beta=${PATH_GATE_BETA:-5.0}
path_strict_gt=${PATH_STRICT_GT:-True}
path_max_remaining_actions=${PATH_MAX_REMAINING_ACTIONS:-20}
path_alignment_mode=${PATH_ALIGNMENT_MODE:-full_start}
path_include_full_path=${PATH_INCLUDE_FULL_PATH:-True}
trajectory_index_path=${TRAJECTORY_INDEX_PATH:-examples/smrc_sd/data/alfworld_verified_walkthrough_prefixes.jsonl}
processed_data_dir=${PROCESSED_DATA_DIR:-data/processed/verl-agent}
# Compatibility-only switch for passing algorithm.sdar.* fields. This is not the
# algorithmic SDAR ablation knob; use PATH_LOSS_TYPE=sdl to remove SDAR gating.
include_sdar_config=${INCLUDE_SDAR_CONFIG:-True}

sdar_args=()
if [[ "$include_sdar_config" == "True" || "$include_sdar_config" == "true" || "$include_sdar_config" == "1" ]]; then
    sdar_args=(
        "+algorithm.sdar.sdar_coef=$path_opd_coef"
        "+algorithm.sdar.gate_beta=$path_gate_beta"
        "+algorithm.sdar.skills_dir=$task_skills_dir"
        "+algorithm.sdar.skill_all=false"
    )
fi

val_rollout_n=${VAL_ROLLOUT_N:-1}
val_env_parallelism=${VAL_ENV_PARALLELISM:-0}

environment_args=(
    "env.env_name=$task_env_name"
    "env.seed=$env_seed"
    "env.max_steps=$max_steps"
    "env.history_length=$env_history_length"
    "env.rollout.n=$group_size"
    "env.resources_per_worker.num_cpus=$num_cpus_per_env_worker"
)
if [[ "${task_env_name,,}" == *"webshop"* ]]; then
    environment_args+=(
        "env.webshop.use_small=$webshop_use_small"
        "env.webshop.human_goals=$webshop_human_goals"
        "env.webshop.oracle_query_variant=$webshop_oracle_query_variant"
        "env.webshop.goal_order_seed=$webshop_goal_order_seed"
        "env.webshop.validation_goal_seed=$webshop_validation_goal_seed"
        "env.webshop.val_parallelism=$val_env_parallelism"
        "env.webshop.oracle_manifest_path=$webshop_oracle_manifest_path"
        "env.webshop.validation_manifest_path=$webshop_validation_manifest_path"
    )
    if [[ -n "$webshop_require_think_tags" ]]; then
        environment_args+=("env.webshop.require_think_tags=$webshop_require_think_tags")
    fi
else
    environment_args+=(
        "env.alfworld.prompt_style=$alfworld_prompt_style"
        "env.alfworld.val_parallelism=$val_env_parallelism"
        "+env.alfworld.penalize_extra_output_after_action=$penalize_extra_output_after_action"
        "+env.alfworld.train_game_files=$alfworld_train_game_files"
    )
    if [[ -n "$alfworld_require_think_tags" ]]; then
        environment_args+=("env.alfworld.require_think_tags=$alfworld_require_think_tags")
    fi
fi

runtime_args=()
[[ -n "${DATA_SEED:-}" ]] && runtime_args+=("+data.seed=$DATA_SEED")
[[ -n "${ROLLOUT_SEED:-}" ]] && runtime_args+=("actor_rollout_ref.rollout.seed=$ROLLOUT_SEED")
[[ -n "${GENERATION_SEED:-}" ]] && runtime_args+=("actor_rollout_ref.rollout.generation_seed=$GENERATION_SEED")
[[ -n "${RESUME_MODE:-}" ]] && runtime_args+=("trainer.resume_mode=$RESUME_MODE")
[[ -n "${CHECKPOINT_DIR:-}" ]] && runtime_args+=("trainer.default_local_dir=$CHECKPOINT_DIR")

python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size \
    --local_dir "$processed_data_dir"

python3 -m "$trainer_main_module" \
    algorithm.adv_estimator=grpo \
    algorithm.path_opd.enable=$path_opd_enable \
    algorithm.path_opd.teacher_context=$path_teacher_context \
    algorithm.path_opd.loss_type=$path_loss_type \
    algorithm.path_opd.opd_coef=$path_opd_coef \
    algorithm.path_opd.sdl_sample_filter=$path_sdl_sample_filter \
    algorithm.path_opd.sdl_token_scope=$path_sdl_token_scope \
    algorithm.path_opd.sdl_mask_special_tokens=$path_sdl_mask_special_tokens \
    algorithm.path_opd.sdl_loss_normalization=$path_sdl_loss_normalization \
    algorithm.path_opd.sdl_loss_mode=$path_sdl_loss_mode \
    algorithm.path_opd.state_sdl_selection=$path_state_sdl_selection \
    algorithm.path_opd.state_match_context=$path_state_match_context \
    algorithm.path_opd.recovery_prompt_style=$path_recovery_prompt_style \
    algorithm.path_opd.recovery_guidance_style=$path_recovery_guidance_style \
    +algorithm.path_opd.enable_walkthrough_trace_recovery=$path_enable_walkthrough_trace_recovery \
    algorithm.path_opd.recovery_match_mode=$path_recovery_match_mode \
    algorithm.path_opd.recovery_state_domain=$path_recovery_state_domain \
    algorithm.path_opd.recovery_alfworld_object_match_mode=$path_recovery_alfworld_object_match_mode \
    algorithm.path_opd.recovery_webshop_match_mode=$path_recovery_webshop_match_mode \
    algorithm.path_opd.gate_beta=$path_gate_beta \
    algorithm.path_opd.trajectory_index_path=$trajectory_index_path \
    algorithm.path_opd.strict_gt_path=$path_strict_gt \
    algorithm.path_opd.max_remaining_actions=$path_max_remaining_actions \
    algorithm.path_opd.alignment_mode=$path_alignment_mode \
    algorithm.path_opd.include_full_path=$path_include_full_path \
    data.train_files="$processed_data_dir/text/train.parquet" \
    data.val_files="$processed_data_dir/text/test.parquet" \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_activation_offload=$enable_activation_offload \
    actor_rollout_ref.actor.strategy=$actor_strategy \
    actor_rollout_ref.ref.strategy=$ref_strategy \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.loss_agg_mode=$loss_agg_mode \
    actor_rollout_ref.actor.use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ppo_max_token_len_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=$actor_use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$actor_kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=$actor_kl_loss_type \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=$actor_param_offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$actor_optimizer_offload \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$log_prob_max_token_len_per_gpu \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.temperature=$rollout_temperature \
    actor_rollout_ref.rollout.top_p=$rollout_top_p \
    actor_rollout_ref.rollout.top_k=$rollout_top_k \
    actor_rollout_ref.rollout.gpu_memory_utilization=$rollout_gpu_memory_utilization \
    actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens \
    actor_rollout_ref.rollout.enable_chunked_prefill=$rollout_enable_chunked_prefill \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=$rollout_free_cache_engine \
    actor_rollout_ref.rollout.keep_engine_awake_during_multiturn=$rollout_keep_engine_awake_during_multiturn \
    actor_rollout_ref.rollout.val_kwargs.temperature=$val_temperature \
    actor_rollout_ref.rollout.val_kwargs.do_sample=$val_do_sample \
    actor_rollout_ref.rollout.val_kwargs.n=$val_rollout_n \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$log_prob_max_token_len_per_gpu \
    actor_rollout_ref.ref.fsdp_config.param_offload=$ref_param_offload \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    "${sdar_args[@]}" \
    "${environment_args[@]}" \
    trainer.critic_warmup=0 \
    "trainer.logger=$logger" \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.save_freq=$save_freq \
    trainer.max_actor_ckpt_to_keep=$max_actor_ckpt_to_keep \
    trainer.max_critic_ckpt_to_keep=$max_critic_ckpt_to_keep \
    trainer.test_freq=$test_freq \
    trainer.total_epochs=$total_epochs \
    trainer.total_training_steps=$total_training_steps \
    trainer.val_before_train=$val_before_train \
    trainer.log_val_generations=$log_val_generations \
    trainer.rollout_data_dir=$rollout_data_dir \
    trainer.rollout_dump_freq=$rollout_dump_freq \
    +trainer.rollout_only=$rollout_only \
    trainer.validation_data_dir=$validation_data_dir \
    "${runtime_args[@]}" \
    "$@"
