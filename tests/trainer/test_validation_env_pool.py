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

from types import SimpleNamespace

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from agent_system.environments.env_package.alfworld.envs import AlfworldEnvs


class _FakeValEnvs:
    validation_capacity = 2

    def prepare_validation_batch(self, batch):
        batch.non_tensor_batch["env_kwargs"] = np.array(
            [{"gamefile": f"game-{item_id}"} for item_id in batch.non_tensor_batch["validation_item_id"]],
            dtype=object,
        )
        return batch


class _FakeCollector:
    def __init__(self):
        self.calls = []

    def multi_turn_loop(self, *, gen_batch, **_kwargs):
        self.calls.append(list(gen_batch.non_tensor_batch["validation_item_id"]))
        chunk_index = len(self.calls) - 1
        return DataProto.from_single_dict(
            {
                "value": gen_batch.batch["value"].clone(),
                "validation_item_id": np.array(gen_batch.non_tensor_batch["validation_item_id"], dtype=object),
                "env_kwargs": np.array(gen_batch.non_tensor_batch["env_kwargs"], dtype=object),
                "success_rate": np.full(len(gen_batch), float(chunk_index), dtype=np.float32),
            }
        )


def test_validation_rollout_uses_bounded_env_waves_and_global_success_rate():
    trainer = object.__new__(RayPPOTrainer)
    trainer.val_envs = _FakeValEnvs()
    trainer.traj_collector = _FakeCollector()
    trainer.actor_rollout_wg = SimpleNamespace()
    trainer.global_steps = 7
    batch = DataProto.from_single_dict(
        {
            "value": torch.arange(4),
            "validation_item_id": np.array(["0:0", "0:0", "0:1", "0:1"], dtype=object),
        }
    )

    output = trainer._run_validation_rollout(batch)

    assert trainer.traj_collector.calls == [["0:0", "0:0"], ["0:1", "0:1"]]
    assert output.batch["value"].tolist() == [0, 1, 2, 3]
    assert output.non_tensor_batch["success_rate"].tolist() == [0.5, 0.5, 0.5, 0.5]
    assert [item["gamefile"] for item in output.non_tensor_batch["env_kwargs"]] == ["game-0:0", "game-0:0", "game-0:1", "game-0:1"]


def test_alfworld_validation_replicas_map_to_the_same_fixed_game():
    envs = object.__new__(AlfworldEnvs)
    envs.is_train = False
    envs.fixed_game_files = ["game-a", "game-b"]

    reset_kwargs = envs.validation_reset_kwargs(["0:0", "0:0", "0:1", "0:1"])

    assert [item["gamefile"] for item in reset_kwargs] == ["game-a", "game-a", "game-b", "game-b"]
    assert all(item["strict_gamefile"] for item in reset_kwargs)
