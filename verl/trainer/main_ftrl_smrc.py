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

"""FTRL main-table entry: plain FullPath-SD or exact-state SMRC-SD."""

import hydra
from omegaconf import OmegaConf


def validate_contract(config):
    expected = {
        "env.env_name": "ftrl",
        "algorithm.adv_estimator": "grpo",
        "algorithm.path_opd.loss_type": "sdl",
        "algorithm.path_opd.sdl_loss_mode": "chosen_token_k3",
        "algorithm.path_opd.sdl_loss_normalization": "response_token_mean",
        "algorithm.path_opd.student_context": "none",
        "algorithm.path_opd.policy_loss_filter": "none",
        "algorithm.path_opd.sdl_mask_special_tokens": False,
        "algorithm.path_opd.in_batch_success_path": False,
        "actor_rollout_ref.actor.use_kl_loss": False,
        "actor_rollout_ref.actor.use_invalid_action_penalty": False,
        "algorithm.use_kl_in_reward": False,
        "actor_rollout_ref.rollout.n": 1,
    }
    for key, value in expected.items():
        if OmegaConf.select(config, key) != value:
            raise ValueError(f"FTRL contract mismatch: {key}")
    variant = config.env.ftrl.variant
    if variant not in ("fullpath", "exact"):
        raise ValueError("FTRL release entry accepts only fullpath or exact")
    if not config.env.ftrl.references or not config.env.ftrl.state_defaults:
        raise ValueError("Set env.ftrl.references and env.ftrl.state_defaults")
    if not config.data.train_files or not config.data.val_files:
        raise ValueError("Set data.train_files and data.val_files")
    expected_selection = "all" if variant == "fullpath" else "match_only"
    if config.algorithm.path_opd.state_sdl_selection != expected_selection:
        raise ValueError("Incorrect SDL routing for the selected FTRL variant")
    view = config.env.ftrl.get("teacher_view", "auto")
    if view not in ("auto", "plain_fullpath") or (variant == "exact" and view == "plain_fullpath"):
        raise ValueError("Invalid FTRL teacher view")


@hydra.main(config_path="config", config_name="ftrl_smrc", version_base=None)
def main(config):
    from verl.trainer.main_path_opd import run_path_opd

    validate_contract(config)
    run_path_opd(config)


if __name__ == "__main__":
    main()
