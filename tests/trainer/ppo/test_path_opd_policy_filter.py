# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import pytest
import torch

from verl.trainer.ppo.path_privileged_ray_trainer import PathPrivOPDRayTrainer
from verl.trainer.ppo.path_privileged_utils import PathPrivilegedContextProvider
from verl.workers.actor.dp_actor import _apply_policy_loss_sample_weight


def test_policy_loss_sample_weight_applies_without_multi_turn_mask():
    response_mask = torch.ones(3, 4)
    sample_weight = torch.tensor([1.0, 0.0, 0.5])

    weighted_mask = _apply_policy_loss_sample_weight(response_mask, sample_weight)

    torch.testing.assert_close(weighted_mask[0], torch.ones(4))
    torch.testing.assert_close(weighted_mask[1], torch.zeros(4))
    torch.testing.assert_close(weighted_mask[2], torch.full((4,), 0.5))


def test_policy_loss_sample_weight_clamps_negative_values():
    weighted_mask = _apply_policy_loss_sample_weight(torch.ones(2, 2), torch.tensor([-1.0, 1.0]))
    torch.testing.assert_close(weighted_mask, torch.tensor([[0.0, 0.0], [1.0, 1.0]]))


def test_policy_loss_sample_weight_rejects_batch_mismatch():
    with pytest.raises(ValueError, match="batch_size"):
        _apply_policy_loss_sample_weight(torch.ones(2, 3), torch.ones(3))


def test_trainer_records_policy_filter_metrics_without_overwriting_loss_mask():
    trainer = object.__new__(PathPrivOPDRayTrainer)
    trainer._last_teacher_skill_metrics = {}
    original_loss_mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    batch = SimpleNamespace(
        batch={
            "responses": torch.zeros(2, 2, dtype=torch.long),
            "response_mask": torch.ones(2, 2),
            "loss_mask": original_loss_mask.clone(),
        }
    )

    trainer._record_policy_loss_sample_weight_metrics(batch, torch.tensor([1.0, 0.0]))

    torch.testing.assert_close(batch.batch["loss_mask"], original_loss_mask)
    assert trainer._last_teacher_skill_metrics["path_opd/policy_loss_sample_weight_nonzero_ratio"] == 0.5
    assert trainer._last_teacher_skill_metrics["path_opd/policy_loss_token_ratio"] == 0.5


def test_d11_none_filter_keeps_grpo_while_d12_repeat_filter_drops_tail():
    provider = object.__new__(PathPrivilegedContextProvider)
    provider.policy_loss_repeat_drop_weight = 0.0
    repeat_metrics = {"path_opd/sdl_repeat_drop_ratio": 1.0}

    provider.policy_loss_filter = "none"
    assert provider._policy_loss_sample_weight(repeat_metrics) == 1.0

    provider.policy_loss_filter = "repeat_drop"
    assert provider._policy_loss_sample_weight(repeat_metrics) == 0.0
