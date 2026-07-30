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

import json

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.rlsd_ray_trainer import build_teacher_batch
from verl.trainer.ppo.rlsd_utils import SkillProvider


def _make_skill_provider(tmp_path):
    mapping = {
        "skill_files": {
            "general_skills": "general.md",
            "cool": "cool.md",
            "pick_and_place": "pick_and_place.md",
        },
        "task_to_skill": {
            "pick_cool_then_place_in_recep": "cool",
            "pick_two_obj_and_place": "pick_and_place",
            "pick_and_place": "pick_and_place",
        },
        "task_keywords": {
            "pick_cool_then_place_in_recep": ["cool"],
            "pick_two_obj_and_place": ["put two"],
            "pick_and_place": ["put"],
        },
    }
    (tmp_path / "skill_mapping.json").write_text(json.dumps(mapping))
    (tmp_path / "general.md").write_text("GENERAL")
    (tmp_path / "cool.md").write_text("COOL")
    (tmp_path / "pick_and_place.md").write_text("PICK")
    return SkillProvider(str(tmp_path))


def test_prompt_skill_inference_prefers_first_specific_match(tmp_path):
    provider = _make_skill_provider(tmp_path)

    assert provider.get_privileged_info_from_prompt("cool some pan and put it in stoveburner") == "GENERAL\n\nCOOL"
    assert provider.get_privileged_info_from_prompt("put two spraybottle in toilet") == "GENERAL\n\nPICK"
    assert provider.get_privileged_info_from_prompt("look around") == "GENERAL"


def test_text_data_source_routes_to_prompt_keyword_matching(tmp_path):
    """Upstream SDAR PR #36: parquet data_source='text' must use prompt skills."""
    provider = _make_skill_provider(tmp_path)

    assert provider.get_privileged_info_from_data_source(
        "text", "cool some pan and put it in stoveburner"
    ) == "GENERAL\n\nCOOL"
    assert provider.get_privileged_info_from_data_source("text", "look around") == "GENERAL"


def _make_webshop_any_skill_provider(tmp_path):
    mapping = {
        "skill_files": {
            "general_skills": "general.md",
            "apparel": "apparel.md",
            "beauty_health": "beauty.md",
            "other": "other.md",
        },
        "task_to_skill": {
            "apparel": "apparel",
            "beauty_health": "beauty_health",
            "other": "other",
        },
        "task_keyword_match": "any",
        "task_keywords": {
            "apparel": ["shirt", "dress", "pants"],
            "beauty_health": ["shampoo", "hair", "cream"],
            "other": [],
        },
    }
    (tmp_path / "skill_mapping.json").write_text(json.dumps(mapping))
    (tmp_path / "general.md").write_text("GENERAL")
    (tmp_path / "apparel.md").write_text("APPAREL")
    (tmp_path / "beauty.md").write_text("BEAUTY")
    (tmp_path / "other.md").write_text("OTHER")
    return SkillProvider(str(tmp_path))


def test_webshop_any_keyword_match_loads_category_and_other_fallback(tmp_path):
    provider = _make_webshop_any_skill_provider(tmp_path)

    assert provider.task_keyword_match == "any"
    assert provider.get_privileged_info_from_prompt("Find me a red dress under 50") == "GENERAL\n\nAPPAREL"
    assert provider.get_privileged_info_from_prompt("Find me hair loss products") == "GENERAL\n\nBEAUTY"
    assert provider.get_privileged_info_from_data_source("text", "Find me a red dress") == "GENERAL\n\nAPPAREL"
    # Unmatched WebShop goals fall back to the explicit empty-keyword "other" bucket.
    assert provider.get_privileged_info_from_prompt("Find me camping gear") == "GENERAL\n\nOTHER"


def test_alfworld_all_match_unchanged_by_webshop_any_mode(tmp_path):
    """Default task_keyword_match=all must keep ALFWorld AND semantics."""
    provider = _make_skill_provider(tmp_path)
    assert provider.task_keyword_match == "all"
    # Single-token OR lists must NOT falsely fire under ALFWorld AND mode.
    assert provider.infer_task_type_from_prompt("I see a dress on the table") is None
    assert provider.get_privileged_info_from_prompt("put two spraybottle in toilet") == "GENERAL\n\nPICK"


class _FakeTokenizer:
    pad_token_id = 0

    def decode(self, token_ids, skip_special_tokens=False):
        return "<|im_start|>user\nYour task is to: cool some pan and put it in stoveburner."

    def encode(self, text, add_special_tokens=False):
        return list(range(1, len(text.split()) + 1))


def _make_teacher_source_batch(gamefile):
    return DataProto.from_dict(
        tensors={
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "responses": torch.tensor([[3]]),
        },
        non_tensors={
            "data_source": np.array(["text"], dtype=object),
            "gamefile": np.array([gamefile], dtype=object),
        },
    )


def test_teacher_batch_records_gamefile_skill_route(tmp_path):
    provider = _make_skill_provider(tmp_path)
    batch = _make_teacher_source_batch("/tmp/pick_cool_then_place_in_recep/trial/game.tw-pddl")

    teacher_batch = build_teacher_batch(batch, provider, _FakeTokenizer(), max_prompt_length=128)

    assert teacher_batch.meta_info["teacher_skill_metrics"] == {
        "teacher_skill/samples": 1.0,
        "teacher_skill/route_gamefile_ratio": 1.0,
        "teacher_skill/task_pick_cool_then_place_in_recep_ratio": 1.0,
    }


def test_teacher_batch_falls_back_to_prompt_when_gamefile_is_missing(tmp_path):
    provider = _make_skill_provider(tmp_path)
    batch = _make_teacher_source_batch(None)

    teacher_batch = build_teacher_batch(batch, provider, _FakeTokenizer(), max_prompt_length=128)

    assert teacher_batch.meta_info["teacher_skill_metrics"] == {
        "teacher_skill/samples": 1.0,
        "teacher_skill/route_prompt_fallback_ratio": 1.0,
        "teacher_skill/task_pick_cool_then_place_in_recep_ratio": 1.0,
    }
