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

from agent_system.environments.env_package.alfworld.projection import alfworld_projection


def test_alfworld_projection_can_accept_qwen3_no_thinking_response():
    responses = ["I should move to the cabinet.\n\n<action>go to cabinet 1</action>"]

    actions, valids = alfworld_projection(responses, [[]], require_think_tags=False)

    assert actions == ["go to cabinet 1"]
    assert valids == [1]


def test_alfworld_projection_keeps_strict_think_check_when_enabled():
    responses = ["I should move to the cabinet.\n\n<action>go to cabinet 1</action>"]

    actions, valids = alfworld_projection(responses, [[]], require_think_tags=True)

    assert actions == ["go to cabinet 1"]
    assert valids == [0]


def test_alfworld_projection_accepts_full_think_action_format():
    responses = ["<think>I should move to the cabinet.</think>\n<action>go to cabinet 1</action>"]

    actions, valids = alfworld_projection(responses, [[]], require_think_tags=True)

    assert actions == ["go to cabinet 1"]
    assert valids == [1]
