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

from agent_system.environments.env_package.webshop.projection import webshop_projection
from agent_system.environments.env_manager import WebshopEnvironmentManager
from omegaconf import OmegaConf


def test_webshop_projection_can_accept_qwen3_no_thinking_response():
    responses = ["I should search for the requested item.\n\n<action>search[red mug]</action>"]

    actions, valids = webshop_projection(responses, require_think_tags=False)

    assert actions == ["search[red mug]"]
    assert valids == [1]


def test_webshop_projection_keeps_strict_think_check_when_enabled():
    responses = ["I should search for the requested item.\n\n<action>search[red mug]</action>"]

    actions, valids = webshop_projection(responses, require_think_tags=True)

    assert actions == ["search[red mug]"]
    assert valids == [0]


def test_webshop_projection_accepts_full_think_action_format():
    responses = ["<think>I should search for the item.</think>\n<action>search[red mug]</action>"]

    actions, valids = webshop_projection(responses, require_think_tags=True)

    assert actions == ["search[red mug]"]
    assert valids == [1]


def test_webshop_no_thinking_config_uses_prompt_without_think_tag_requirement():
    config = OmegaConf.create(
        {
            "data": {"apply_chat_template_kwargs": {"enable_thinking": False}},
            "env": {"history_length": 2, "webshop": {"require_think_tags": None}},
        }
    )
    manager = WebshopEnvironmentManager(envs=None, projection_f=None, config=config)
    manager.tasks = ["buy a red mug"]

    prompts = manager.build_text_obs(
        ["Search page"],
        [{"available_actions": {"has_search_bar": True, "clickables": []}}],
        init=True,
    )

    assert manager.require_think_tags is False
    assert "MUST be enclosed within <think> </think> tags" not in prompts[0]
    assert "present it within <action> </action> tags" in prompts[0]
