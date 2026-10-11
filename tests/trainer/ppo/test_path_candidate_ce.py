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

import torch

from verl import DataProto
from verl.trainer.ppo.path_privileged_utils import build_path_candidate_ce_batch


class _CharacterTokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(character) + 1 for character in text]

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        values = token_ids.tolist() if isinstance(token_ids, torch.Tensor) else list(token_ids)
        return "".join(chr(int(token_id) - 1) for token_id in values if int(token_id) > 0)


class _MatcherProvider:
    state_sdl_selection = "match_only"
    state_sdl_random_seed = 0
    policy_loss_filter = "none"

    def start_batch_metrics(self):
        self._sample_metrics = []
        self._sample_contexts = []
        self._sample_candidate_actions = []

    def build_context(self, *, batch, idx, prompt_text):
        del batch, prompt_text
        matched = float(idx == 0)
        self._sample_metrics.append(
            {
                "path_opd/recovery_state_match_ratio": matched,
                "path_opd/sdl_sample_weight_mean": 1.0,
            }
        )
        self._sample_contexts.append(f"privileged-{idx}")
        self._sample_candidate_actions.append(("go north", "open door")[idx])
        return self._sample_contexts[-1]

    def finish_batch_metrics(self):
        return {"path_opd/samples": float(len(self._sample_metrics))}

    @staticmethod
    def _policy_loss_sample_weight(_sample_metrics):
        return 1.0


def test_candidate_ce_uses_original_prompt_and_match_only_candidate_target():
    tokenizer = _CharacterTokenizer()
    prompt_ids = torch.tensor([[0, 0, 66, 67], [0, 0, 68, 69]], dtype=torch.long)
    prompt_mask = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=torch.long)
    sampled_responses = torch.tensor([[80, 81, 0], [82, 83, 84]], dtype=torch.long)
    sampled_response_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.long)
    batch = DataProto.from_dict(
        tensors={
            "input_ids": torch.cat([prompt_ids, sampled_responses], dim=-1),
            "attention_mask": torch.cat([prompt_mask, sampled_response_mask], dim=-1),
            "position_ids": torch.zeros(2, 7, dtype=torch.long),
            "responses": sampled_responses,
            "response_mask": sampled_response_mask,
        }
    )

    candidate_batch = build_path_candidate_ce_batch(
        batch=batch,
        context_provider=_MatcherProvider(),
        tokenizer=tokenizer,
        max_response_length=64,
        global_step=7,
    )

    torch.testing.assert_close(candidate_batch.batch["input_ids"][:, :4], prompt_ids)
    assert tokenizer.decode(
        candidate_batch.batch["responses"][0][
            candidate_batch.batch["candidate_response_mask"][0].bool()
        ]
    ) == "<action>go north</action>"
    assert tokenizer.decode(
        candidate_batch.batch["responses"][1][
            candidate_batch.batch["candidate_response_mask"][1].bool()
        ]
    ) == "<action>open door</action>"
    torch.testing.assert_close(
        candidate_batch.batch["candidate_sample_weight"],
        torch.tensor([1.0, 0.0]),
    )
    torch.testing.assert_close(
        candidate_batch.batch["normalization_token_count"],
        torch.tensor([2.0, 3.0]),
    )
    assert batch.non_tensor_batch["path_candidate_ce_action"].tolist() == [
        "go north",
        "open door",
    ]
    assert (
        candidate_batch.meta_info["candidate_ce_metrics"][
            "path_opd/candidate_ce_selected_sample_ratio"
        ]
        == 0.5
    )
