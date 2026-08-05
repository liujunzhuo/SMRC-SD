# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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

import pytest

from agent_system.environments.env_package.alfworld.envs import select_eval_game_files


def test_eval_game_selection_is_unique_and_reproducible():
    game_files = [f"game-{idx}" for idx in range(140)]

    first = select_eval_game_files(game_files, env_num=128, seed=1000)
    second = select_eval_game_files(list(reversed(game_files)), env_num=128, seed=1000)

    assert first == second
    assert len(first) == 128
    assert len(set(first)) == 128


def test_eval_game_selection_rejects_oversized_request():
    with pytest.raises(ValueError, match="Requested 4"):
        select_eval_game_files(["game-0", "game-1", "game-2"], env_num=4, seed=1000)
