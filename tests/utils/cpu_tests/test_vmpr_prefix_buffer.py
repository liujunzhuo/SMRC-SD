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
from collections import defaultdict
from types import SimpleNamespace

import numpy as np

from agent_system.multi_turn_rollout.prefix_buffer import ASSISTED_SOURCE, STUDENT_SOURCE, TEACHER_SOURCE, PrefixBuffer
from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector


def test_empty_prefix_buffer_falls_back_to_full_start():
    buffer = PrefixBuffer(env_name="alfworld/AlfredTWEnv")

    assert len(buffer) == 0
    assert buffer.sample(global_step=0) is None


def test_seed_prefixes_are_teacher_seed(tmp_path):
    seed_path = tmp_path / "teacher_seed.jsonl"
    seed_path.write_text(
        json.dumps(
            {
                "prefix_id": "seed-1",
                "env_name": "alfworld/AlfredTWEnv",
                "task_key": "game.tw-pddl",
                "reset_kwargs": {"gamefile": "game.tw-pddl"},
                "prefix_actions": ["look", "open fridge"],
                "boundary_type": "step_2",
                "prefix_len": 2,
                "insert_step": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    buffer = PrefixBuffer(env_name="alfworld/AlfredTWEnv", seed_prefix_path=str(seed_path))
    entry = buffer.sample(global_step=3)

    assert entry is not None
    assert entry.source == TEACHER_SOURCE
    assert entry.to_reset_kwargs()["gamefile"] == "game.tw-pddl"
    assert entry.to_reset_kwargs()["prefix_actions"] == ["look", "open fridge"]


def test_student_and_assisted_sources_do_not_mix():
    buffer = PrefixBuffer(env_name="alfworld/AlfredTWEnv", step_boundaries=[1, 3, 8])

    student_added = buffer.add_trajectory_prefixes(
        source=STUDENT_SOURCE,
        env_name="alfworld/AlfredTWEnv",
        task_key="student-game",
        reset_kwargs={"gamefile": "student-game"},
        actions=["look", "go north", "take apple", "put apple"],
        global_step=10,
    )
    assisted_added = buffer.add_trajectory_prefixes(
        source=ASSISTED_SOURCE,
        env_name="alfworld/AlfredTWEnv",
        task_key="assisted-game",
        reset_kwargs={"gamefile": "assisted-game"},
        actions=["open fridge", "take milk", "close fridge", "put milk"],
        global_step=11,
    )

    counts = buffer.source_counts()
    assert student_added == 2
    assert assisted_added == 2
    assert counts[STUDENT_SOURCE] == 2
    assert counts[ASSISTED_SOURCE] == 2
    assert counts[TEACHER_SOURCE] == 0


def test_replay_result_updates_verified_and_fail_count():
    buffer = PrefixBuffer(env_name="alfworld/AlfredTWEnv", step_boundaries=[1])
    buffer.add_trajectory_prefixes(
        source=STUDENT_SOURCE,
        env_name="alfworld/AlfredTWEnv",
        task_key="game",
        reset_kwargs={"gamefile": "game"},
        actions=["look", "go north"],
        global_step=1,
    )
    entry = next(iter(buffer.entries.values()))

    buffer.mark_replay_result(entry.prefix_id, success=False)
    assert buffer.entries[entry.prefix_id].replay_fail_count == 1
    assert not buffer.entries[entry.prefix_id].verified

    buffer.mark_replay_result(entry.prefix_id, success=True)
    assert buffer.entries[entry.prefix_id].verified


def test_online_prefix_survives_when_teacher_seed_buffer_is_full(tmp_path):
    seed_path = tmp_path / "teacher_seed.jsonl"
    rows = []
    for idx in range(2):
        rows.append(
            {
                "prefix_id": f"seed-{idx}",
                "env_name": "alfworld/AlfredTWEnv",
                "task_key": f"seed-game-{idx}",
                "reset_kwargs": {"gamefile": f"seed-game-{idx}"},
                "prefix_actions": ["look"],
                "boundary_type": "step_1",
                "prefix_len": 1,
                "insert_step": 0,
            }
        )
    seed_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    buffer = PrefixBuffer(env_name="alfworld/AlfredTWEnv", seed_prefix_path=str(seed_path), max_buffer_size=2, step_boundaries=[1])
    buffer.add_trajectory_prefixes(
        source=STUDENT_SOURCE,
        env_name="alfworld/AlfredTWEnv",
        task_key="student-game",
        reset_kwargs={"gamefile": "student-game"},
        actions=["look", "go north"],
        global_step=10,
    )

    counts = buffer.source_counts()
    assert len(buffer) == 2
    assert counts[STUDENT_SOURCE] == 1
    assert counts[TEACHER_SOURCE] == 1


def test_prefer_longest_below_max_prefix():
    buffer = PrefixBuffer(env_name="alfworld/AlfredTWEnv", step_boundaries=[2, 4, 6])
    buffer.add_trajectory_prefixes(
        source=TEACHER_SOURCE,
        env_name="alfworld/AlfredTWEnv",
        task_key="teacher-game",
        reset_kwargs={"gamefile": "teacher-game"},
        actions=["a1", "a2", "a3", "a4", "a5", "a6", "a7"],
        global_step=0,
    )

    sampled = {buffer.sample(global_step=step, max_prefix_len=5, prefer_longest_below_max=True).prefix_len for step in range(20)}
    assert sampled == {4}


def test_prefix_sampling_is_reproducible_with_private_seed():
    def make_buffer(seed):
        buffer = PrefixBuffer(
            env_name="alfworld/AlfredTWEnv",
            step_boundaries=[1, 2, 3],
            source_sampling_strategy="source_first",
            teacher_source_weight=1.0,
            student_source_weight=1.0,
            seed=seed,
        )
        for source, task_key in ((TEACHER_SOURCE, "teacher-game"), (STUDENT_SOURCE, "student-game")):
            buffer.add_trajectory_prefixes(
                source=source,
                env_name="alfworld/AlfredTWEnv",
                task_key=task_key,
                reset_kwargs={"gamefile": task_key},
                actions=["a1", "a2", "a3", "a4"],
                global_step=0,
            )
        return buffer

    first = make_buffer(seed=17)
    second = make_buffer(seed=17)
    first_sequence = [(entry.source, entry.prefix_len, entry.task_key) for step in range(20) if (entry := first.sample(global_step=step)) is not None]
    second_sequence = [(entry.source, entry.prefix_len, entry.task_key) for step in range(20) if (entry := second.sample(global_step=step)) is not None]

    assert first_sequence == second_sequence


def test_prefix_sample_exposes_pre_update_diagnostics():
    buffer = PrefixBuffer(env_name="alfworld/AlfredTWEnv", step_boundaries=[1], seed=3)
    buffer.add_trajectory_prefixes(
        source=STUDENT_SOURCE,
        env_name="alfworld/AlfredTWEnv",
        task_key="student-game",
        reset_kwargs={"gamefile": "student-game"},
        actions=["look", "go north"],
        global_step=4,
    )

    entry = buffer.sample(global_step=10)
    stats = buffer.last_sample_stats

    assert entry is not None
    assert stats is not None
    assert stats["task_key"] == "student-game"
    assert stats["age"] == 6
    assert stats["idle_steps"] == 6
    assert stats["use_count_before"] == 0
    assert entry.use_count == 1
    assert entry.last_used_step == 10


def test_student_reuse_limit_does_not_remove_teacher_prefix():
    buffer = PrefixBuffer(
        env_name="alfworld/AlfredTWEnv",
        step_boundaries=[1],
        teacher_source_weight=0.0,
        student_source_weight=1.0,
        max_student_prefix_use_count=2,
    )
    buffer.add_trajectory_prefixes(
        source=TEACHER_SOURCE,
        env_name="alfworld/AlfredTWEnv",
        task_key="teacher-game",
        reset_kwargs={"gamefile": "teacher-game"},
        actions=["look", "go north"],
        global_step=0,
    )
    buffer.add_trajectory_prefixes(
        source=STUDENT_SOURCE,
        env_name="alfworld/AlfredTWEnv",
        task_key="student-game",
        reset_kwargs={"gamefile": "student-game"},
        actions=["look", "go south"],
        global_step=1,
    )

    assert buffer.sample(global_step=2).source == STUDENT_SOURCE
    assert buffer.sample(global_step=3).source == STUDENT_SOURCE
    assert buffer.source_counts()[STUDENT_SOURCE] == 0
    assert buffer.source_counts()[TEACHER_SOURCE] == 1


def test_student_prefix_length_ratio_limits_only_student_boundaries():
    buffer = PrefixBuffer(
        env_name="alfworld/AlfredTWEnv",
        step_boundaries=[1, 2, 3, 4, 5],
        max_student_prefix_length_ratio=0.5,
    )
    actions = [f"action-{idx}" for idx in range(7)]

    student_added = buffer.add_trajectory_prefixes(
        source=STUDENT_SOURCE,
        env_name="alfworld/AlfredTWEnv",
        task_key="student-game",
        reset_kwargs={"gamefile": "student-game"},
        actions=actions,
        global_step=1,
    )
    teacher_added = buffer.add_trajectory_prefixes(
        source=TEACHER_SOURCE,
        env_name="alfworld/AlfredTWEnv",
        task_key="teacher-game",
        reset_kwargs={"gamefile": "teacher-game"},
        actions=actions,
        global_step=1,
    )

    student_lengths = sorted(entry.prefix_len for entry in buffer.entries.values() if entry.source == STUDENT_SOURCE)
    teacher_lengths = sorted(entry.prefix_len for entry in buffer.entries.values() if entry.source == TEACHER_SOURCE)
    assert student_added == 3
    assert student_lengths == [1, 2, 3]
    assert teacher_added == 5
    assert teacher_lengths == [1, 2, 3, 4, 5]


def test_prefix_buffer_deduplicates_exact_cleaned_path_boundary_only():
    buffer = PrefixBuffer(env_name="alfworld/AlfredTWEnv", step_boundaries=[1, 2])
    common = {
        "source": STUDENT_SOURCE,
        "env_name": "alfworld/AlfredTWEnv",
        "task_key": "student-game",
        "reset_kwargs": {"gamefile": "student-game"},
        "global_step": 1,
    }

    assert buffer.add_trajectory_prefixes(actions=["go north", "take apple", "put apple"], **common) == 2
    assert buffer.add_trajectory_prefixes(actions=[" go  north ", "TAKE APPLE", "put apple"], **common) == 0
    assert buffer.last_add_duplicate_count == 2
    assert buffer.add_trajectory_prefixes(actions=["go south", "take apple", "put apple"], **common) == 2
    assert len(buffer) == 4


def test_group_success_ratio_above_threshold_skips_student_admission():
    collector = TrajectoryCollector.__new__(TrajectoryCollector)
    collector.vmpr_enabled = True
    collector.vmpr_cfg = {
        "skip_add_when_group_success_ratio_above": 0.5,
        "update_student_prefixes": True,
        "update_assisted_prefixes": False,
        "save_every_steps": 0,
    }
    collector.config = SimpleNamespace(env=SimpleNamespace(env_name="alfworld/AlfredTWEnv"))
    collector.prefix_buffer = PrefixBuffer(env_name="alfworld/AlfredTWEnv", step_boundaries=[1])
    collector._vmpr_pending_metrics = defaultdict(float)
    collector.vmpr_success_max_length_ratio_to_canonical = 0.0
    collector.vmpr_success_max_action_len = 0

    trajectories = []
    infos = []
    for idx in range(8):
        trajectories.append(
            [
                {
                    "uid": "group-1",
                    "traj_uid": f"traj-{idx}",
                    "vmpr_is_prefix_start": False,
                    "vmpr_source": "full_start",
                    "active_masks": True,
                    "executed_action": f"action-{idx}",
                },
                {
                    "uid": "group-1",
                    "traj_uid": f"traj-{idx}",
                    "vmpr_is_prefix_start": False,
                    "vmpr_source": "full_start",
                    "active_masks": True,
                    "executed_action": "finish",
                },
            ]
        )
        infos.append([{"extra.gamefile": f"game-{idx}"}])

    collector._vmpr_update_prefix_buffer(
        total_batch_list=trajectories,
        total_infos=infos,
        episode_rewards=np.ones(8),
        success={"success_rate": np.array([1, 1, 1, 1, 1, 0, 0, 0])},
        global_step=1,
        is_train=True,
    )

    assert len(collector.prefix_buffer) == 0
    assert collector._vmpr_pending_metrics["vmpr/skipped_group_high_success_trajectories"] == 5


def test_student_insertion_freshness_is_not_refreshed_by_last_use():
    buffer = PrefixBuffer(
        env_name="alfworld/AlfredTWEnv",
        student_insertion_freshness_decay=0.1,
        freshness_decay=0.001,
    )
    old_entry = buffer.add_trajectory_prefixes(
        source=STUDENT_SOURCE,
        env_name="alfworld/AlfredTWEnv",
        task_key="old",
        reset_kwargs={"gamefile": "old"},
        actions=["look", "go north"],
        global_step=0,
    )
    assert old_entry == 1
    old = next(iter(buffer.entries.values()))
    old.last_used_step = 99

    buffer.add_trajectory_prefixes(
        source=STUDENT_SOURCE,
        env_name="alfworld/AlfredTWEnv",
        task_key="new",
        reset_kwargs={"gamefile": "new"},
        actions=["look", "go south"],
        global_step=90,
    )
    new = next(entry for entry in buffer.entries.values() if entry.task_key == "new")

    old_weight = buffer._entry_sampling_weight(old, global_step=100, include_source_weight=False)
    new_weight = buffer._entry_sampling_weight(new, global_step=100, include_source_weight=False)
    assert new_weight > old_weight
    assert old_weight == 1 / 11
    assert new_weight == 1 / 2


def test_group_best_admission_keeps_shortest_successful_trajectory():
    collector = TrajectoryCollector.__new__(TrajectoryCollector)
    collector.vmpr_enabled = True
    collector.vmpr_cfg = {
        "group_best_success_only": True,
        "skip_add_when_group_success_all": True,
        "update_student_prefixes": True,
        "update_assisted_prefixes": False,
        "save_every_steps": 0,
    }
    collector.config = SimpleNamespace(env=SimpleNamespace(env_name="alfworld/AlfredTWEnv"))
    collector.prefix_buffer = PrefixBuffer(env_name="alfworld/AlfredTWEnv", step_boundaries=[1])
    collector._vmpr_pending_metrics = defaultdict(float)

    def trajectory(actions):
        return [
            {
                "uid": "group-1",
                "traj_uid": f"traj-{len(actions)}",
                "vmpr_is_prefix_start": False,
                "vmpr_source": "full_start",
                "active_masks": True,
                "executed_action": action,
            }
            for action in actions
        ]

    trajectories = [
        trajectory(["a1", "a2", "a3"]),
        trajectory(["b1", "b2"]),
        trajectory(["c1", "c2", "c3", "c4"]),
    ]
    infos = [[{"extra.gamefile": f"game-{idx}"}] for idx in range(3)]
    collector._vmpr_update_prefix_buffer(
        total_batch_list=trajectories,
        total_infos=infos,
        episode_rewards=np.array([1.0, 1.0, 0.0]),
        success={"success_rate": np.array([1.0, 1.0, 0.0])},
        global_step=10,
        is_train=True,
    )

    assert len(collector.prefix_buffer) == 1
    entry = next(iter(collector.prefix_buffer.entries.values()))
    assert entry.metadata["full_actions"] == ["b1", "b2"]
    assert collector._vmpr_pending_metrics["vmpr/skipped_non_best_success_trajectories"] == 1


def test_collector_reports_sampled_prefix_quality_and_coverage():
    collector = TrajectoryCollector.__new__(TrajectoryCollector)
    collector._vmpr_pending_metrics = {}
    collector._vmpr_add_sampled_prefix_metrics(
        [
            (
                {
                    "source": TEACHER_SOURCE,
                    "task_key": "teacher-game",
                    "age": 10,
                    "idle_steps": 3,
                    "use_count_before": 2,
                    "replay_fail_count": 0,
                    "success_ema": 0.75,
                    "verified": True,
                },
                4,
            ),
            (
                {
                    "source": STUDENT_SOURCE,
                    "task_key": "student-game",
                    "age": 2,
                    "idle_steps": 1,
                    "use_count_before": 0,
                    "replay_fail_count": 1,
                    "success_ema": 0.5,
                    "verified": False,
                },
                4,
            ),
        ]
    )

    assert collector._vmpr_pending_metrics["vmpr/sampled_prefix_age_mean"] == 6
    assert collector._vmpr_pending_metrics["vmpr/sampled_prefix_unique_tasks"] == 2
    assert collector._vmpr_pending_metrics["vmpr/sampled_prefix_unique_task_ratio"] == 1
    assert collector._vmpr_pending_metrics["vmpr/sampled_prefix_teacher_seed_verified_ratio"] == 1
    assert collector._vmpr_pending_metrics["vmpr/sampled_prefix_student_autonomous_replay_fail_count_mean"] == 1


def test_static_vmpr_prefix_len_bounds_apply_without_schedule():
    collector = TrajectoryCollector.__new__(TrajectoryCollector)
    collector.vmpr_cfg = {
        "min_prefix_len": 1,
        "max_prefix_len": 4,
        "prefix_len_schedule": None,
    }
    collector.config = SimpleNamespace(env=SimpleNamespace(max_steps=30))
    collector._vmpr_pending_metrics = {}

    assert collector._vmpr_prefix_len_bounds(global_step=100) == (1, 4)


def test_success_path_cleaner_drops_prompt_non_admissible_action():
    trajectory = [
        {
            "active_masks": True,
            "turn_step": 0,
            "vmpr_current_observation": "You are in the middle of a room.",
            "vmpr_next_observation": "You arrive at diningtable 1.",
            "vmpr_admissible_actions": ["go to diningtable 1"],
            "executed_action": "go to diningtable 1",
            "is_action_valid": True,
        },
        {
            "active_masks": True,
            "turn_step": 1,
            "vmpr_current_observation": "You arrive at diningtable 1. On it, you see bread 1.",
            "vmpr_next_observation": "Nothing happens.",
            "vmpr_admissible_actions": ["go to countertop 1", "take bread 1 from diningtable 1"],
            "executed_action": "take potato 1 from diningtable 1",
            "is_action_valid": True,
        },
        {
            "active_masks": True,
            "turn_step": 2,
            "vmpr_current_observation": "Nothing happens.",
            "vmpr_next_observation": "You arrive at countertop 1.",
            "vmpr_admissible_actions": ["go to countertop 1"],
            "executed_action": "go to countertop 1",
            "is_action_valid": True,
        },
    ]

    actions, dropped, duplicates, no_progress, non_admissible = TrajectoryCollector._vmpr_clean_trajectory_actions(trajectory)

    assert actions == ["go to diningtable 1", "go to countertop 1"]
    assert dropped == 0
    assert duplicates == 0
    assert no_progress == 1
    assert non_admissible == 1


def test_success_path_cleaner_keeps_webshop_search_against_placeholder():
    long_query = "A" * 150
    trajectory = [
        {
            "active_masks": True,
            "turn_step": 0,
            "vmpr_current_observation": "Search home",
            "vmpr_next_observation": "Search results",
            "vmpr_admissible_actions": ["search[<your query>]", "click[back to search]"],
            "executed_action": f"search[{long_query}]",
            "is_action_valid": True,
        },
        {
            "active_masks": True,
            "turn_step": 1,
            "vmpr_current_observation": "Search results",
            "vmpr_next_observation": "Item page",
            "vmpr_admissible_actions": ["click[B012345678]", "click[back to search]"],
            "executed_action": "click[B012345678]",
            "is_action_valid": True,
        },
        {
            "active_masks": True,
            "turn_step": 2,
            "vmpr_current_observation": "Item page",
            "vmpr_next_observation": "Thank you",
            "vmpr_admissible_actions": ["click[buy now]", "click[blue]"],
            "executed_action": "click[not-admissible]",
            "is_action_valid": True,
        },
    ]

    actions, dropped, duplicates, no_progress, non_admissible = TrajectoryCollector._vmpr_clean_trajectory_actions(trajectory)

    assert actions == [f"search[{long_query}]", "click[B012345678]"]
    assert dropped == 0
    assert duplicates == 0
    assert no_progress == 0
    assert non_admissible == 1


def test_success_path_cleaner_still_uses_exact_match_for_alfworld_lists():
    # Guard against WebShop search helpers leaking into ALFWorld cleanup.
    trajectory = [
        {
            "active_masks": True,
            "turn_step": 0,
            "vmpr_current_observation": "You are in the middle of a room.",
            "vmpr_next_observation": "You arrive at desk 1.",
            "vmpr_admissible_actions": ["go to desk 1", "inventory"],
            "executed_action": "go to desk 1",
            "is_action_valid": True,
        },
        {
            "active_masks": True,
            "turn_step": 1,
            "vmpr_current_observation": "You arrive at desk 1.",
            "vmpr_next_observation": "Nothing happens.",
            "vmpr_admissible_actions": ["take pencil 1 from desk 1", "inventory"],
            "executed_action": "take pencil 2 from desk 1",
            "is_action_valid": True,
        },
    ]

    actions, dropped, duplicates, no_progress, non_admissible = TrajectoryCollector._vmpr_clean_trajectory_actions(trajectory)

    assert actions == ["go to desk 1"]
    assert non_admissible == 1
    assert no_progress == 1
