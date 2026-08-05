# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main entry point for SMRC-SD / FullPath-SD training."""

import os

import hydra
import ray
from omegaconf import OmegaConf


def path_opd_uses_sdl_sample_weights(path_cfg) -> bool:
    return (
        bool(path_cfg.get("sdl_confidence_weighting", False))
        or float(path_cfg.get("recovery_success_confirmation_sdl_weight", 1.0)) != 1.0
        or str(path_cfg.get("state_sdl_selection", "all") or "all").lower() != "all"
    )


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_path_opd(config)


def run_path_opd(config) -> None:
    if not ray.is_initialized():
        from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print("[SMRC-SD] Starting local Ray runtime.")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    runner = PathOPDTaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class PathOPDTaskRunner:
    def run(self, config):
        from omegaconf import OmegaConf, open_dict

        from verl.utils.fs import copy_to_local

        if os.environ.get("VERBOSE_CONFIG", "0") == "1":
            from pprint import pprint

            pprint(OmegaConf.to_container(config, resolve=True))
        else:
            print("[SMRC-SD] Configuration resolved (set VERBOSE_CONFIG=1 to print it).")
        OmegaConf.resolve(config)

        path_cfg = config.algorithm.get("path_opd", {})
        sdar_cfg = config.algorithm.get("sdar", {})
        loss_type = str(path_cfg.get("loss_type", "sdar")).lower()
        with open_dict(config):
            if loss_type == "sdar":
                config.actor_rollout_ref.actor.use_sdar_loss = True
                config.actor_rollout_ref.actor.sdar_loss_coef = path_cfg.get("opd_coef", sdar_cfg.get("sdar_coef", 0.1))
                config.actor_rollout_ref.actor.sdar_gate_beta = path_cfg.get("gate_beta", sdar_cfg.get("gate_beta", 5.0))
                config.actor_rollout_ref.actor.sdar_loss_mask_special_tokens = path_cfg.get("sdl_mask_special_tokens", False)
            elif loss_type == "sdl":
                sdl_loss_mode = str(path_cfg.get("sdl_loss_mode", "chosen_token_k3") or "chosen_token_k3").lower()
                supported_sdl_modes = {
                    "chosen_token_k3",
                    "k3",
                    "topk",
                    "topk_forward_kl",
                    "teacher_topk_forward_kl",
                    "forward_kl_topk",
                }
                if sdl_loss_mode not in supported_sdl_modes:
                    raise ValueError(
                        f"Unsupported algorithm.path_opd.sdl_loss_mode={sdl_loss_mode!r}; "
                        f"expected one of {sorted(supported_sdl_modes)}"
                    )
                use_topk_sdl = sdl_loss_mode in {
                    "topk",
                    "topk_forward_kl",
                    "teacher_topk_forward_kl",
                    "forward_kl_topk",
                }
                if use_topk_sdl and config.actor_rollout_ref.actor.strategy not in ("fsdp", "fsdp2"):
                    raise ValueError("Path-OPD Top-k SDL currently requires actor strategy fsdp or fsdp2")
                if use_topk_sdl and bool(config.actor_rollout_ref.actor.get("use_fused_kernels", False)):
                    raise ValueError("Path-OPD Top-k SDL requires actor.use_fused_kernels=False")
                sdl_topk = int(path_cfg.get("sdl_topk", 32))
                sdl_topk_chunk_size = int(path_cfg.get("sdl_topk_chunk_size", 256))
                if use_topk_sdl and sdl_topk <= 0:
                    raise ValueError(f"algorithm.path_opd.sdl_topk must be positive, got {sdl_topk}")
                if use_topk_sdl and sdl_topk_chunk_size <= 0:
                    raise ValueError(
                        f"algorithm.path_opd.sdl_topk_chunk_size must be positive, got {sdl_topk_chunk_size}"
                    )
                config.actor_rollout_ref.actor.use_sdl_loss = True
                config.actor_rollout_ref.actor.sdl_loss_coef = path_cfg.get("opd_coef", sdar_cfg.get("sdar_coef", 0.1))
                config.actor_rollout_ref.actor.sdl_loss_sample_filter = path_cfg.get("sdl_sample_filter", "all")
                config.actor_rollout_ref.actor.sdl_loss_token_scope = path_cfg.get("sdl_token_scope", "all")
                config.actor_rollout_ref.actor.sdl_loss_mask_special_tokens = path_cfg.get("sdl_mask_special_tokens", False)
                config.actor_rollout_ref.actor.sdl_loss_sample_weighting = path_opd_uses_sdl_sample_weights(path_cfg)
                config.actor_rollout_ref.actor.sdl_loss_normalization = path_cfg.get("sdl_loss_normalization", "selected_token_mean")
                sdl_topk_loss_clamp = float(path_cfg.get("sdl_topk_loss_clamp", 0.0) or 0.0)
                if use_topk_sdl and sdl_topk_loss_clamp < 0:
                    raise ValueError(
                        f"algorithm.path_opd.sdl_topk_loss_clamp must be non-negative, got {sdl_topk_loss_clamp}"
                    )
                config.actor_rollout_ref.actor.sdl_loss_mode = sdl_loss_mode
                config.actor_rollout_ref.actor.sdl_topk = sdl_topk
                config.actor_rollout_ref.actor.sdl_topk_include_tail = bool(path_cfg.get("sdl_topk_include_tail", True))
                config.actor_rollout_ref.actor.sdl_topk_chunk_size = sdl_topk_chunk_size
                config.actor_rollout_ref.actor.sdl_topk_is_weight = bool(path_cfg.get("sdl_topk_is_weight", False))
                config.actor_rollout_ref.actor.sdl_topk_loss_clamp = sdl_topk_loss_clamp
            elif loss_type == "candidate_ce":
                if config.actor_rollout_ref.actor.strategy not in ("fsdp", "fsdp2"):
                    raise ValueError("Path-OPD candidate CE currently requires actor strategy fsdp or fsdp2")
                candidate_ce_max_response_length = int(
                    path_cfg.get("candidate_ce_max_response_length", 64)
                )
                if candidate_ce_max_response_length <= 0:
                    raise ValueError(
                        "algorithm.path_opd.candidate_ce_max_response_length must be positive, "
                        f"got {candidate_ce_max_response_length}"
                    )
                candidate_ce_normalization = str(
                    path_cfg.get("candidate_ce_normalization", "response_token_mean")
                    or "response_token_mean"
                ).lower()
                supported_candidate_ce_normalizations = {
                    "selected",
                    "selected_token",
                    "selected_token_mean",
                    "response",
                    "response_token",
                    "response_token_mean",
                    "original_response_token_mean",
                }
                if candidate_ce_normalization not in supported_candidate_ce_normalizations:
                    raise ValueError(
                        "Unsupported algorithm.path_opd.candidate_ce_normalization="
                        f"{candidate_ce_normalization!r}; expected selected_token_mean or "
                        "response_token_mean"
                    )
                config.actor_rollout_ref.actor.use_candidate_ce_loss = True
                config.actor_rollout_ref.actor.candidate_ce_loss_coef = path_cfg.get(
                    "opd_coef", sdar_cfg.get("sdar_coef", 0.1)
                )
                config.actor_rollout_ref.actor.candidate_ce_loss_normalization = (
                    candidate_ce_normalization
                )
            else:
                raise ValueError(
                    f"Unsupported algorithm.path_opd.loss_type={loss_type!r}; "
                    "expected 'sdar', 'sdl', or 'candidate_ce'"
                )

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        from agent_system.environments import make_envs

        envs, val_envs = make_envs(config)

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_manager_name = config.reward_model.get("reward_manager", "episode")
        if reward_manager_name == "episode":
            from agent_system.reward_manager import EpisodeRewardManager

            reward_manager_cls = EpisodeRewardManager
        else:
            raise NotImplementedError

        reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, normalize_by_length=False)
        # Keep validation output aggregate-only; generated trajectories are available
        # through the configured JSONL dump when generation logging is enabled.
        val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, normalize_by_length=False)

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        assert config.actor_rollout_ref.rollout.n == 1, (
            "In verl, actor_rollout_ref.rollout.n>1 is for GRPO. "
            "In verl+env, we keep n=1, and achieve GRPO by env.rollout.n"
        )

        from agent_system.multi_turn_rollout import TrajectoryCollector

        traj_collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)

        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        from verl.trainer.ppo.path_privileged_ray_trainer import PathPrivOPDRayTrainer
        from verl.trainer.ppo.path_privileged_utils import PathPrivilegedContextProvider
        from verl.trainer.ppo.rlsd_utils import SkillProvider

        default_skills_dir = "skills/webshop" if "webshop" in str(config.env.env_name).lower() else "skills/alfworld"
        skills_dir = sdar_cfg.get("skills_dir", default_skills_dir)
        skill_all = sdar_cfg.get("skill_all", False)
        skill_provider = SkillProvider(skills_dir=skills_dir, skill_all=skill_all)
        path_context_provider = PathPrivilegedContextProvider(config=config, skill_provider=skill_provider)

        print(f"[Path-OPD] loss_type: {loss_type}")
        print(f"[Path-OPD] teacher_context: {path_cfg.get('teacher_context', 'gt_path')}")
        if loss_type == "candidate_ce":
            print("[Path-OPD] candidate_ce teacher_forward: disabled")
            print(
                "[Path-OPD] candidate_ce target: ordinary student prompt -> "
                "<action>{matcher candidate}</action>"
            )
        print(f"[Path-OPD] trajectory_index_path: {path_cfg.get('trajectory_index_path')}")
        print(f"[Path-OPD] trajectory_index_size: {len(path_context_provider.trajectory_index.by_key)}")
        print(f"[Path-OPD] skills_dir: {skills_dir}")

        trainer = PathPrivOPDRayTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
            traj_collector=traj_collector,
            envs=envs,
            val_envs=val_envs,
            skill_provider=skill_provider,
            path_context_provider=path_context_provider,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
