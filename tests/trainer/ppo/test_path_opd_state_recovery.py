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
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from verl.trainer.main_path_opd import path_opd_uses_sdl_sample_weights
from verl.trainer.ppo.path_privileged_utils import PathPrivilegedContextProvider, WalkthroughState, build_state_sdl_selection_mask


def _prompt(observation, admissible):
    actions = "\n".join(f" '{action}'" for action in admissible)
    return (
        "user\n\nYour task is to: examine the pillow with the desklamp.\n"
        f"You are now at step 5 and your current observation is: {observation}\n"
        f"Your admissible actions of the current situation are: [{actions}].\n\n"
        "Now it's your turn to take an action.\n\nassistant\n<think>\n\n</think>\n\n"
    )


class _TraceCache:
    def __init__(self, states):
        self.states = states

    def get(self, _gamefile, _full_actions):
        return self.states


class _FailIfCalledTraceCache:
    def get(self, _gamefile, _full_actions):
        raise AssertionError("structured-state recovery must not construct a TextWorld trace")


def _state_provider(events, trace_state):
    provider = object.__new__(PathPrivilegedContextProvider)
    provider.recovery_match_mode = "state"
    provider.recovery_use_last_progress_observation_on_no_progress = False
    provider.recovery_correction_only = False
    provider.recovery_skip_if_action_matches_student = False
    provider.enable_walkthrough_trace_recovery = True
    provider.max_remaining_actions = 20
    provider.trace_cache = _TraceCache([trace_state])
    provider._trajectory_prefix_events = lambda _batch, _idx: list(events)
    provider._trajectory_prefix_actions = lambda _batch, _idx: [event["action"] for event in events]
    return provider


def test_state_sdl_selection_masks_use_hard_match_metric():
    sample_metrics = [
        {"path_opd/recovery_state_match_ratio": 1.0},
        {"path_opd/recovery_state_match_ratio": 0.0},
        {"path_opd/recovery_state_match_ratio": 1.0},
        {},
        {"path_opd/recovery_state_match_ratio": 0.0},
    ]

    all_mask, all_metrics = build_state_sdl_selection_mask(sample_metrics, mode="all")
    match_mask, match_metrics = build_state_sdl_selection_mask(sample_metrics, mode="match_only")
    mismatch_mask, mismatch_metrics = build_state_sdl_selection_mask(sample_metrics, mode="mismatch_only")

    assert all_mask == [1.0, 1.0, 1.0, 1.0, 1.0]
    assert match_mask == [1.0, 0.0, 1.0, 0.0, 0.0]
    assert mismatch_mask == [0.0, 1.0, 0.0, 1.0, 1.0]
    assert all_metrics["path_opd/state_sdl_selected_count"] == 5.0
    assert match_metrics["path_opd/state_sdl_selected_count"] == 2.0
    assert mismatch_metrics["path_opd/state_sdl_selected_count"] == 3.0


def test_state_sdl_random_equal_is_exact_per_step_and_deterministic():
    sample_metrics = [
        {"path_opd/recovery_state_match_ratio": float(index in {0, 3, 7, 11})}
        for index in range(16)
    ]

    first_mask, first_metrics = build_state_sdl_selection_mask(
        sample_metrics,
        mode="random_equal",
        random_seed=17,
        global_step=125,
    )
    repeated_mask, repeated_metrics = build_state_sdl_selection_mask(
        sample_metrics,
        mode="random_equal",
        random_seed=17,
        global_step=125,
    )

    assert first_mask == repeated_mask
    assert sum(first_mask) == 4.0
    assert first_metrics["path_opd/state_sdl_match_count"] == 4.0
    assert first_metrics["path_opd/state_sdl_selected_count"] == 4.0
    assert first_metrics["path_opd/state_sdl_selected_minus_match_count"] == 0.0
    assert first_metrics == repeated_metrics


def test_state_sdl_random_equal_handles_zero_and_full_match_budgets():
    no_matches = [{"path_opd/recovery_state_match_ratio": 0.0} for _ in range(4)]
    all_matches = [{"path_opd/recovery_state_match_ratio": 1.0} for _ in range(4)]

    empty_mask, _ = build_state_sdl_selection_mask(no_matches, mode="random_equal", global_step=1)
    full_mask, _ = build_state_sdl_selection_mask(all_matches, mode="random_equal", global_step=1)

    assert empty_mask == [0.0, 0.0, 0.0, 0.0]
    assert full_mask == [1.0, 1.0, 1.0, 1.0]


def test_state_sdl_unsupported_equal_is_exact_disjoint_and_deterministic():
    sample_metrics = [
        {"path_opd/recovery_state_match_ratio": float(index in {0, 3, 7, 11})}
        for index in range(16)
    ]

    first_mask, first_metrics = build_state_sdl_selection_mask(
        sample_metrics,
        mode="unsupported_equal",
        random_seed=17,
        global_step=125,
    )
    repeated_mask, repeated_metrics = build_state_sdl_selection_mask(
        sample_metrics,
        mode="unsupported_equal",
        random_seed=17,
        global_step=125,
    )

    matched_indices = {0, 3, 7, 11}
    selected_indices = {
        index for index, selected in enumerate(first_mask) if selected
    }
    assert first_mask == repeated_mask
    assert len(selected_indices) == len(matched_indices)
    assert selected_indices.isdisjoint(matched_indices)
    assert first_metrics["path_opd/state_sdl_match_count"] == 4.0
    assert first_metrics["path_opd/state_sdl_unsupported_count"] == 12.0
    assert first_metrics["path_opd/state_sdl_selected_count"] == 4.0
    assert first_metrics["path_opd/state_sdl_selected_minus_match_count"] == 0.0
    assert first_metrics["path_opd/state_sdl_selected_match_overlap_count"] == 0.0
    assert (
        first_metrics[
            "path_opd/state_sdl_selected_unsupported_overlap_count"
        ]
        == 4.0
    )
    assert (
        first_metrics["path_opd/state_sdl_selection_unsupported_equal"] == 1.0
    )
    assert first_metrics == repeated_metrics


def test_state_sdl_unsupported_equal_requires_enough_unsupported_rows():
    no_matches = [
        {"path_opd/recovery_state_match_ratio": 0.0} for _ in range(4)
    ]
    zero_budget_mask, _ = build_state_sdl_selection_mask(
        no_matches,
        mode="unsupported_equal",
        global_step=1,
    )
    assert zero_budget_mask == [0.0, 0.0, 0.0, 0.0]

    too_many_matches = [
        {"path_opd/recovery_state_match_ratio": float(index < 3)}
        for index in range(4)
    ]
    with pytest.raises(
        ValueError,
        match="cannot satisfy the exact per-step MatchOnly-sized budget",
    ):
        build_state_sdl_selection_mask(
            too_many_matches,
            mode="unsupported_equal",
            global_step=1,
        )


def test_state_sdl_selection_rejects_invalid_mode():
    with pytest.raises(ValueError, match="Unsupported state SDL selection mode"):
        build_state_sdl_selection_mask([], mode="unknown")


def test_non_all_state_sdl_selection_requires_state_recovery_teacher():
    config = OmegaConf.create(
        {
            "algorithm": {
                "path_opd": {
                    "teacher_context": "gt_path",
                    "recovery_match_mode": "history",
                    "state_sdl_selection": "match_only",
                }
            }
        }
    )

    with pytest.raises(ValueError, match="requires a recovery teacher"):
        PathPrivilegedContextProvider(config=config)


def test_non_all_state_sdl_selection_enables_actor_sample_weighting():
    assert path_opd_uses_sdl_sample_weights({"state_sdl_selection": "match_only"})
    assert path_opd_uses_sdl_sample_weights({"state_sdl_selection": "random_equal"})
    assert path_opd_uses_sdl_sample_weights(
        {"state_sdl_selection": "unsupported_equal"}
    )
    assert not path_opd_uses_sdl_sample_weights({"state_sdl_selection": "all"})


def test_state_match_context_full_path_requires_state_recovery_teacher():
    config = OmegaConf.create(
        {
            "algorithm": {
                "path_opd": {
                    "teacher_context": "gt_path",
                    "recovery_match_mode": "history",
                    "state_match_context": "full_path",
                }
            }
        }
    )

    with pytest.raises(ValueError, match="state_match_context=full_path requires"):
        PathPrivilegedContextProvider(config=config)


def test_state_match_context_rejects_invalid_value():
    config = OmegaConf.create(
        {
            "algorithm": {
                "path_opd": {
                    "teacher_context": "gt_recovery_aligned",
                    "recovery_match_mode": "state",
                    "state_match_context": "bogus",
                }
            }
        }
    )

    with pytest.raises(ValueError, match="Unsupported algorithm.path_opd.state_match_context"):
        PathPrivilegedContextProvider(config=config)


def test_recovery_prompt_style_rejects_invalid_value():
    config = OmegaConf.create(
        {
            "algorithm": {
                "path_opd": {
                    "teacher_context": "gt_recovery_aligned",
                    "recovery_match_mode": "state",
                    "recovery_prompt_style": "verbose_magic",
                }
            }
        }
    )

    with pytest.raises(ValueError, match="Unsupported algorithm.path_opd.recovery_prompt_style"):
        PathPrivilegedContextProvider(config=config)


def test_candidate_only_prompt_requires_alfworld_recovery_context():
    config = OmegaConf.create(
        {
            "algorithm": {
                "path_opd": {
                    "teacher_context": "gt_path",
                    "recovery_match_mode": "history",
                    "recovery_prompt_style": "candidate_only",
                    "state_match_context": "recovery",
                }
            }
        }
    )

    with pytest.raises(ValueError, match="recovery_prompt_style=candidate_only requires"):
        PathPrivilegedContextProvider(config=config)


def test_history_matcher_can_use_sctc_prompt_style():
    """Matcher ablation: history mode + current SCTC payload must construct."""
    config = OmegaConf.create(
        {
            "algorithm": {
                "path_opd": {
                    "teacher_context": "gt_recovery_aligned",
                    "recovery_match_mode": "history",
                    "recovery_prompt_style": "state_summary_candidate",
                    "recovery_guidance_style": "path_on_or_off",
                    "state_match_context": "recovery",
                    "state_sdl_selection": "match_only",
                }
            }
        }
    )
    provider = PathPrivilegedContextProvider(config=config)
    assert provider.recovery_match_mode == "history"
    assert provider.recovery_prompt_style == "state_summary_candidate"
    assert provider.recovery_guidance_style == "path_on_or_off"


def test_state_recovery_matches_same_execution_state_without_prefix_match():
    observation = "You arrive at desk 1. On the desk 1, you see a tissuebox 1."
    events = [
        {"action": "go to bed 1", "next_observation": "You arrive at bed 1."},
        {"action": "take pillow 1 from bed 1", "next_observation": "You pick up the pillow 1."},
        {"action": "go to garbagecan 1", "next_observation": "You arrive at garbagecan 1."},
        {"action": "go to desk 1", "next_observation": observation},
    ]
    trace_state = WalkthroughState(
        step_index=3,
        observation=observation,
        next_observation="You arrive at dresser 1.",
        admissible_actions=["go to dresser 1"],
        next_action="go to dresser 1",
        action_history=["go to bed 1", "take pillow 1 from bed 1", "go to desk 1"],
        observation_norm=observation.lower(),
        next_action_norm="go to dresser 1",
    )
    provider = _state_provider(events, trace_state)
    provider.trace_cache = _FailIfCalledTraceCache()
    batch = SimpleNamespace(
        non_tensor_batch={
            "executed_action": np.array(["go to garbagecan 1"], dtype=object),
            "gamefile": np.array(["game.tw-pddl"], dtype=object),
            "turn_step": np.array([4], dtype=object),
        }
    )
    full_actions = ["go to bed 1", "take pillow 1 from bed 1", "go to desk 1", "go to dresser 1", "use desklamp 1"]

    recovery = provider._recovery_guess(
        batch=batch,
        idx=0,
        prompt_text=_prompt(observation, ["go to dresser 1", "go to bed 1"]),
        full_actions=full_actions,
        consumed=0,
    )

    assert recovery["route"] == "walkthrough_state_compatible"
    assert recovery["action"] == "go to dresser 1"
    assert recovery["use_action"]
    assert recovery["state_match"]
    assert recovery["symbolic_reference"]


def test_best_state_aligned_success_uses_compatible_student_buffer_path(tmp_path):
    index_path = tmp_path / "paths.jsonl"
    index_path.write_text(
        json.dumps(
            {
                "gamefile": "game.tw-pddl",
                "full_actions": [
                    "go to bed 1",
                    "take pillow 1 from bed 1",
                    "go to countertop 1",
                    "go to dresser 1",
                    "use desklamp 1",
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    buffer_path = tmp_path / "buffer.jsonl"
    buffer_path.write_text(
        json.dumps(
            {
                "prefix_id": "student-path-1",
                "source": "student_autonomous",
                "task_key": "game.tw-pddl",
                "insert_step": 10,
                "metadata": {
                    "source_traj_uid": "other-trajectory",
                    "full_actions": [
                        "go to bed 1",
                        "take pillow 1 from bed 1",
                        "go to desk 1",
                        "go to dresser 1",
                        "use desklamp 1",
                    ],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = OmegaConf.create(
        {
            "algorithm": {
                "path_opd": {
                    "teacher_context": "gt_recovery_aligned",
                    "trajectory_index_path": str(index_path),
                    "path_source_policy": "best_state_aligned_success",
                    "path_buffer_path": str(buffer_path),
                    "path_buffer_source_filter": "student_autonomous",
                    "dont_reprompt_on_self_success": True,
                    "alignment_mode": "dynamic",
                    "recovery_match_mode": "state",
                    "enable_walkthrough_trace_recovery": True,
                }
            }
        }
    )
    provider = PathPrivilegedContextProvider(config=config)
    prompts = [
        _prompt("You are in the middle of a room.", ["go to bed 1"]),
        _prompt("You arrive at bed 1.", ["take pillow 1 from bed 1"]),
        _prompt("You pick up the pillow 1 from the bed 1.", ["go to garbagecan 1"]),
        _prompt("You arrive at garbagecan 1.", ["go to desk 1"]),
        _prompt("You arrive at desk 1.", ["go to dresser 1", "go to countertop 1"]),
    ]
    batch = SimpleNamespace(
        non_tensor_batch={
            "traj_uid": np.array(["current-trajectory"] * 5, dtype=object),
            "turn_step": np.arange(5, dtype=object),
            "executed_action": np.array(
                ["go to bed 1", "take pillow 1 from bed 1", "go to garbagecan 1", "go to desk 1", "go to dresser 1"],
                dtype=object,
            ),
            "is_action_valid": np.array([True] * 5, dtype=object),
            "input": np.array(prompts, dtype=object),
            "gamefile": np.array(["game.tw-pddl"] * 5, dtype=object),
        }
    )

    provider.start_batch_metrics()
    context = provider.build_context(batch=batch, idx=4, prompt_text=prompts[4])
    metrics = provider._sample_metrics[-1]

    assert "Your current location, inventory, and task progress match the state after these successful-path actions:" in context
    assert "go to bed 1 -> take pillow 1 from bed 1 -> go to desk 1" in context
    assert "Current state summary:" in context
    assert "This execution state matches" not in context
    assert "Candidate next action for the current state:\ngo to dresser 1" in context
    assert "Use the matched path information as privileged guidance, but reason from the current observation and admissible actions." in context
    assert metrics["path_opd/route_student_autonomous_ratio"] == 1.0
    assert metrics["path_opd/path_selected_state_compatible_buffer_candidate_ratio"] == 1.0
    assert metrics["path_opd/path_state_compatible_buffer_candidate_count_mean"] == 1.0


def _d0matched_index_and_batch(tmp_path):
    index_path = tmp_path / "paths.jsonl"
    index_path.write_text(
        json.dumps(
            {
                "gamefile": "game.tw-pddl",
                "full_actions": [
                    "go to bed 1",
                    "take pillow 1 from bed 1",
                    "go to countertop 1",
                    "go to dresser 1",
                    "use desklamp 1",
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    prompts = [
        _prompt("You are in the middle of a room.", ["go to bed 1"]),
        _prompt("You arrive at bed 1.", ["take pillow 1 from bed 1"]),
        _prompt("You pick up the pillow 1 from the bed 1.", ["go to countertop 1"]),
        _prompt("You arrive at countertop 1.", ["go to dresser 1"]),
        _prompt("You arrive at dresser 1.", ["use desklamp 1", "go to bed 1"]),
    ]
    batch = SimpleNamespace(
        non_tensor_batch={
            "traj_uid": np.array(["current-trajectory"] * 5, dtype=object),
            "turn_step": np.arange(5, dtype=object),
            "executed_action": np.array(
                [
                    "go to bed 1",
                    "take pillow 1 from bed 1",
                    "go to countertop 1",
                    "go to dresser 1",
                    "use desklamp 1",
                ],
                dtype=object,
            ),
            "is_action_valid": np.array([True] * 5, dtype=object),
            "input": np.array(prompts, dtype=object),
            "gamefile": np.array(["game.tw-pddl"] * 5, dtype=object),
        }
    )
    return index_path, prompts, batch


def _d0matched_config(
    index_path,
    *,
    state_match_context,
    state_sdl_selection="match_only",
    recovery_prompt_style="current",
    recovery_guidance_style="style_default",
):
    return OmegaConf.create(
        {
            "algorithm": {
                "path_opd": {
                    "teacher_context": "gt_recovery_aligned",
                    "trajectory_index_path": str(index_path),
                    "alignment_mode": "dynamic",
                    "recovery_match_mode": "state",
                    "enable_walkthrough_trace_recovery": True,
                    "state_sdl_selection": state_sdl_selection,
                    "state_match_context": state_match_context,
                    "recovery_prompt_style": recovery_prompt_style,
                    "recovery_guidance_style": recovery_guidance_style,
                }
            }
        }
    )


def test_state_match_context_full_path_emits_d0_context_but_keeps_match_metric(tmp_path):
    index_path, prompts, batch = _d0matched_index_and_batch(tmp_path)

    # Recovery context (MatchOnly) is the reference: on a matched row it exposes a
    # localized candidate next action.
    recovery_provider = PathPrivilegedContextProvider(config=_d0matched_config(index_path, state_match_context="recovery"))
    recovery_provider.start_batch_metrics()
    recovery_context = recovery_provider.build_context(batch=batch, idx=3, prompt_text=prompts[3])
    recovery_metrics = recovery_provider._sample_metrics[-1]
    assert "Candidate next action for the current state:" in recovery_context
    assert "Your current location, inventory, and task progress match" in recovery_context
    assert "Current state summary:" in recovery_context
    assert float(recovery_metrics.get("path_opd/recovery_state_match_ratio", 0.0)) == 1.0
    assert float(recovery_metrics.get("path_opd/recovery_prompt_style_current", 0.0)) == 1.0

    # D0@Matched: the same matcher runs (so the state-match metric that drives
    # match_only selection is unchanged), but the teacher prompt is the plain
    # full-path (D0) context with no localized candidate or state summary.
    d0_provider = PathPrivilegedContextProvider(config=_d0matched_config(index_path, state_match_context="full_path"))
    d0_provider.start_batch_metrics()
    d0_context = d0_provider.build_context(batch=batch, idx=3, prompt_text=prompts[3])
    d0_metrics = d0_provider._sample_metrics[-1]

    assert "Complete successful path for this task:" in d0_context
    assert "The current state may be on or off this path." in d0_context
    assert "Candidate next action for the current state:" not in d0_context
    assert "Current state summary:" not in d0_context
    assert "Your current location, inventory, and task progress match" not in d0_context
    # Row selection signal must survive so that match_only picks the same rows.
    assert float(d0_metrics.get("path_opd/recovery_state_match_ratio", 0.0)) == 1.0

    selection_mask, _ = build_state_sdl_selection_mask([d0_metrics], mode="match_only")
    assert selection_mask == [1.0]


def test_default_history_recovery_prompt_keeps_d2r_prefix_renderer(tmp_path):
    index_path, prompts, batch = _d0matched_index_and_batch(tmp_path)
    config = _d0matched_config(index_path, state_match_context="recovery")
    config.algorithm.path_opd.recovery_match_mode = "history"
    config.algorithm.path_opd.state_sdl_selection = "all"
    config.algorithm.path_opd.alignment_mode = "full_start"
    provider = PathPrivilegedContextProvider(config=config)
    provider._recovery_guess = lambda **_kwargs: {
        "route": "walkthrough_trace_obs_match_history_suffix",
        "action": "go to dresser 1",
        "continuation": ["go to dresser 1", "use desklamp 1"],
        "confidence": 1.0,
        "use_action": True,
        "path_start_index": 3,
        "trace_history_len": 3,
        "inventory_match": True,
        "match_basis": "history",
    }
    provider.start_batch_metrics()

    context = provider.build_context(batch=batch, idx=3, prompt_text=prompts[3])

    assert "Matched prefix:" in context
    assert "go to bed 1 -> take pillow 1 from bed 1 -> go to countertop 1" in context
    assert "Candidate next action for the current state:" in context
    assert "Use the matched path information as privileged guidance" in context
    assert "Current state summary:" not in context


def test_history_matcher_sctc_match_only_keeps_payload_and_selects_hits(tmp_path):
    index_path, prompts, batch = _d0matched_index_and_batch(tmp_path)
    provider = PathPrivilegedContextProvider(
        config=_d0matched_config(
            index_path,
            state_match_context="recovery",
            recovery_prompt_style="state_summary_candidate",
            recovery_guidance_style="path_on_or_off",
        )
    )
    provider.recovery_match_mode = "history"
    provider._recovery_guess = lambda **_kwargs: {
        "route": "walkthrough_trace_obs_match_history_suffix",
        "action": "go to dresser 1",
        "continuation": ["go to dresser 1", "use desklamp 1"],
        "confidence": 1.0,
        "use_action": True,
        "state_match": True,
        "match_basis": "history",
        "path_start_index": 3,
        "trace_history_len": 3,
        "inventory_match": True,
        "state_signature": {
            "location": "countertop 1",
            "inventory": ("pillow 1",),
            "object_locations": (("pillow 1", "inventory"),),
            "object_properties": (),
            "activated_tools": (),
        },
    }
    provider.start_batch_metrics()

    context = provider.build_context(batch=batch, idx=3, prompt_text=prompts[3])
    metrics = provider._sample_metrics[-1]

    assert "Complete successful path for this task:" in context
    assert "Current state summary:" in context
    assert "location=countertop 1" in context
    assert "Candidate next action for the current state:" in context
    assert "go to dresser 1" in context
    assert "The current state may be on or off this path." in context
    assert "Matched prefix:" not in context
    assert "Use the matched path information as privileged guidance" not in context
    assert float(metrics["path_opd/recovery_state_match_ratio"]) == 1.0

    selection_mask, _ = build_state_sdl_selection_mask([metrics], mode="match_only")
    assert selection_mask == [1.0]


def test_d0_random_equal_uses_full_path_context_and_match_count_budget(tmp_path):
    index_path, prompts, batch = _d0matched_index_and_batch(tmp_path)
    provider = PathPrivilegedContextProvider(
        config=_d0matched_config(
            index_path,
            state_match_context="full_path",
            state_sdl_selection="random_equal",
        )
    )
    provider.start_batch_metrics()

    contexts = [
        provider.build_context(batch=batch, idx=index, prompt_text=prompt)
        for index, prompt in enumerate(prompts)
    ]
    sample_metrics = list(provider._sample_metrics)
    selection_mask, selection_metrics = build_state_sdl_selection_mask(
        sample_metrics,
        mode=provider.state_sdl_selection,
        random_seed=provider.state_sdl_random_seed,
        global_step=125,
    )

    assert all("Complete successful path for this task:" in context for context in contexts)
    assert all("The current state may be on or off this path." in context for context in contexts)
    assert all("Candidate next action for the current state:" not in context for context in contexts)
    assert all("Current state summary:" not in context for context in contexts)
    assert sum(selection_mask) == selection_metrics["path_opd/state_sdl_match_count"]
    assert selection_metrics["path_opd/state_sdl_selected_minus_match_count"] == 0.0


def test_d0_unsupported_equal_uses_full_path_context_and_disjoint_budget(
    tmp_path,
):
    index_path, prompts, batch = _d0matched_index_and_batch(tmp_path)
    provider = PathPrivilegedContextProvider(
        config=_d0matched_config(
            index_path,
            state_match_context="full_path",
            state_sdl_selection="unsupported_equal",
        )
    )
    provider.start_batch_metrics()

    contexts = [
        provider.build_context(batch=batch, idx=index, prompt_text=prompt)
        for index, prompt in enumerate(prompts)
    ]
    sample_metrics = list(provider._sample_metrics)
    # This fixture intentionally contains only supported states. Flip three
    # selector labels after context construction so the integration check can
    # verify that full-path rendering is independent of the disjoint route.
    for index in range(2, len(sample_metrics)):
        sample_metrics[index] = dict(sample_metrics[index])
        sample_metrics[index]["path_opd/recovery_state_match_ratio"] = 0.0
    selection_mask, selection_metrics = build_state_sdl_selection_mask(
        sample_metrics,
        mode=provider.state_sdl_selection,
        random_seed=provider.state_sdl_random_seed,
        global_step=125,
    )
    match_mask = [
        float(metrics["path_opd/recovery_state_match_ratio"] > 0.5)
        for metrics in sample_metrics
    ]

    assert all(
        "Complete successful path for this task:" in context
        for context in contexts
    )
    assert all(
        "The current state may be on or off this path." in context
        for context in contexts
    )
    assert all(
        "Candidate next action for the current state:" not in context
        for context in contexts
    )
    assert all("Current state summary:" not in context for context in contexts)
    assert sum(selection_mask) == sum(match_mask)
    assert not any(
        selected and matched
        for selected, matched in zip(selection_mask, match_mask)
    )
    assert selection_metrics["path_opd/state_sdl_selected_minus_match_count"] == 0.0
    assert (
        selection_metrics[
            "path_opd/state_sdl_selected_match_overlap_count"
        ]
        == 0.0
    )


def test_candidate_only_state_recovery_omits_reference_prefix_and_state_summary(tmp_path):
    index_path, prompts, batch = _d0matched_index_and_batch(tmp_path)
    provider = PathPrivilegedContextProvider(
        config=_d0matched_config(
            index_path,
            state_match_context="recovery",
            recovery_prompt_style="candidate_only",
        )
    )
    provider.start_batch_metrics()

    context = provider.build_context(batch=batch, idx=3, prompt_text=prompts[3])
    metrics = provider._sample_metrics[-1]

    assert "Complete successful path for this task:" in context
    assert "Candidate next action for the current state:" in context
    assert "Use the path information as privileged guidance" in context
    assert "Use the matched path information as privileged guidance" not in context
    assert "Matched prefix:" not in context
    assert "Successful-path actions before this state:" not in context
    assert "Your current location, inventory, and task progress match" not in context
    assert "Current state summary:" not in context
    assert float(metrics["path_opd/recovery_state_match_ratio"]) == 1.0
    assert float(metrics["path_opd/recovery_candidate_matches_sampled_ratio"]) == 1.0
    assert float(metrics["path_opd/recovery_prompt_style_candidate_only"]) == 1.0
    assert float(metrics["path_opd/recovery_prompt_style_current"]) == 0.0

    selection_mask, _ = build_state_sdl_selection_mask([metrics], mode="match_only")
    assert selection_mask == [1.0]


def test_candidate_prefix_state_recovery_keeps_reference_prefix_but_omits_state_summary(tmp_path):
    index_path, prompts, batch = _d0matched_index_and_batch(tmp_path)
    provider = PathPrivilegedContextProvider(
        config=_d0matched_config(
            index_path,
            state_match_context="recovery",
            recovery_prompt_style="candidate_prefix",
        )
    )
    provider.start_batch_metrics()

    context = provider.build_context(batch=batch, idx=3, prompt_text=prompts[3])
    metrics = provider._sample_metrics[-1]

    assert "Complete successful path for this task:" in context
    assert "Candidate next action for the current state:" in context
    assert "Your current location, inventory, and task progress match" in context
    assert "Use the matched path information as privileged guidance" in context
    assert "Current state summary:" not in context
    assert float(metrics["path_opd/recovery_state_match_ratio"]) == 1.0
    assert float(metrics["path_opd/recovery_candidate_matches_sampled_ratio"]) == 1.0
    assert float(metrics["path_opd/recovery_prompt_style_candidate_prefix"]) == 1.0
    assert float(metrics["path_opd/recovery_prompt_style_candidate_only"]) == 0.0
    assert float(metrics["path_opd/recovery_prompt_style_current"]) == 0.0

    selection_mask, _ = build_state_sdl_selection_mask([metrics], mode="match_only")
    assert selection_mask == [1.0]


def test_state_summary_only_recovery_omits_reference_prefix_and_candidate(tmp_path):
    index_path, prompts, batch = _d0matched_index_and_batch(tmp_path)
    provider = PathPrivilegedContextProvider(
        config=_d0matched_config(
            index_path,
            state_match_context="recovery",
            recovery_prompt_style="state_summary_only",
        )
    )
    provider.start_batch_metrics()

    context = provider.build_context(batch=batch, idx=3, prompt_text=prompts[3])
    metrics = provider._sample_metrics[-1]

    assert "Complete successful path for this task:" in context
    assert "Current state summary:" in context
    assert (
        "location=countertop 1; inventory=['pillow 1']; "
        "object_locations={'pillow 1': 'inventory'}; object_properties=[]"
    ) in context
    assert "activated_tools=" not in context
    assert "current location=" not in context
    assert "Use the path and current state summary as privileged guidance" in context
    assert "The current state may be on or off this path." not in context
    assert "Candidate next action for the current state:" not in context
    assert "Your current location, inventory, and task progress match" not in context
    assert "Successful-path actions before this state:" not in context
    assert "Matched prefix:" not in context
    assert provider._sample_candidate_actions[-1] == "go to dresser 1"
    assert float(metrics["path_opd/recovery_state_match_ratio"]) == 1.0
    # The candidate remains internal to the matcher/selection path, but is not
    # rendered in the teacher prompt.
    assert float(metrics["path_opd/recovery_prompt_style_state_summary_only"]) == 1.0
    assert float(metrics["path_opd/recovery_prompt_style_candidate_only"]) == 0.0
    assert float(metrics["path_opd/recovery_prompt_style_candidate_prefix"]) == 0.0
    assert float(metrics["path_opd/recovery_prompt_style_current"]) == 0.0

    selection_mask, _ = build_state_sdl_selection_mask([metrics], mode="match_only")
    assert selection_mask == [1.0]


def test_state_summary_candidate_recovery_omits_reference_prefix(tmp_path):
    index_path, prompts, batch = _d0matched_index_and_batch(tmp_path)
    provider = PathPrivilegedContextProvider(
        config=_d0matched_config(
            index_path,
            state_match_context="recovery",
            recovery_prompt_style="state_summary_candidate",
        )
    )
    provider.start_batch_metrics()

    context = provider.build_context(batch=batch, idx=3, prompt_text=prompts[3])
    metrics = provider._sample_metrics[-1]

    assert "Complete successful path for this task:" in context
    assert "Current state summary:" in context
    assert (
        "location=countertop 1; inventory=['pillow 1']; "
        "object_locations={'pillow 1': 'inventory'}; object_properties=[]"
    ) in context
    assert "activated_tools=" not in context
    assert "current location=" not in context
    assert "Candidate next action for the current state:" in context
    assert "go to dresser 1" in context
    assert "Use the matched path information as privileged guidance" in context
    assert "The current state may be on or off this path." not in context
    assert "Your current location, inventory, and task progress match" not in context
    assert "Successful-path actions before this state:" not in context
    assert "Matched prefix:" not in context
    assert float(metrics["path_opd/recovery_state_match_ratio"]) == 1.0
    assert float(metrics["path_opd/recovery_prompt_style_state_summary_candidate"]) == 1.0
    assert float(metrics["path_opd/recovery_prompt_style_state_summary_only"]) == 0.0
    assert float(metrics["path_opd/recovery_prompt_style_candidate_only"]) == 0.0
    assert float(metrics["path_opd/recovery_prompt_style_candidate_prefix"]) == 0.0
    assert float(metrics["path_opd/recovery_prompt_style_current"]) == 0.0
    assert float(metrics["path_opd/recovery_guidance_style_path_on_or_off"]) == 0.0

    selection_mask, _ = build_state_sdl_selection_mask([metrics], mode="match_only")
    assert selection_mask == [1.0]


def test_state_summary_candidate_path_on_or_off_guidance_keeps_payload(tmp_path):
    index_path, prompts, batch = _d0matched_index_and_batch(tmp_path)
    provider = PathPrivilegedContextProvider(
        config=_d0matched_config(
            index_path,
            state_match_context="recovery",
            recovery_prompt_style="state_summary_candidate",
            recovery_guidance_style="path_on_or_off",
        )
    )
    provider.start_batch_metrics()

    context = provider.build_context(batch=batch, idx=3, prompt_text=prompts[3])
    metrics = provider._sample_metrics[-1]

    assert "Complete successful path for this task:" in context
    assert "Current state summary:" in context
    assert "Candidate next action for the current state:" in context
    assert "go to dresser 1" in context
    assert "The current state may be on or off this path." in context
    assert "Use the path as privileged guidance, but reason from the current observation and admissible actions." in context
    assert "Use the matched path information as privileged guidance" not in context
    assert "Your current location, inventory, and task progress match" not in context
    assert float(metrics["path_opd/recovery_prompt_style_state_summary_candidate"]) == 1.0
    assert float(metrics["path_opd/recovery_guidance_style_path_on_or_off"]) == 1.0


def test_state_signature_text_omits_activated_tools_only():
    """Teacher render drops activated_tools; all other one-line fields stay intact."""
    signature = {
        "location": "microwave 1",
        "inventory": ("mug 1",),
        "object_locations": (("mug 1", "inventory"), ("apple 1", "countertop 1")),
        "object_properties": ("heat:mug 1",),
        "activated_tools": ("desklamp 1",),
    }

    text = PathPrivilegedContextProvider._state_signature_text(signature)

    assert text == (
        "location=microwave 1; inventory=['mug 1']; "
        "object_locations={'mug 1': 'inventory', 'apple 1': 'countertop 1'}; "
        "object_properties=['heat:mug 1']"
    )
    assert "activated_tools" not in text
    assert "current location=" not in text
    assert "\n" not in text

    # Canonical matcher storage / heat-cool transitions stay op-first.
    events = [
        {"action": "cool mug 1 with fridge 1", "next_observation": "You cool the mug 1 using the fridge 1."},
        {"action": "heat mug 1 with microwave 1", "next_observation": "You heat the mug 1 using the microwave 1."},
    ]
    heated = PathPrivilegedContextProvider._execution_state_signature(events, "", {"mug 1"})
    assert heated["object_properties"] == ("heat:mug 1",)
    assert "object_properties=['heat:mug 1']" in PathPrivilegedContextProvider._state_signature_text(heated)


def test_execution_state_temperature_tracks_latest_transition():
    events = [
        {"action": "cool mug 1 with fridge 1", "next_observation": "You cool the mug 1 using the fridge 1."},
        {"action": "heat mug 1 with microwave 1", "next_observation": "You heat the mug 1 using the microwave 1."},
    ]

    heated = PathPrivilegedContextProvider._execution_state_signature(events, "", {"mug 1"})
    cooled = PathPrivilegedContextProvider._execution_state_signature(list(reversed(events)), "", {"mug 1"})

    assert heated["object_properties"] == ("heat:mug 1",)
    assert cooled["object_properties"] == ("cool:mug 1",)


def test_execution_state_location_prefers_latest_successful_navigation():
    events = [
        {"action": "go to drawer 6", "next_observation": "You arrive at drawer 6."},
        {"action": "open safe 1", "next_observation": "You open the safe 1. The safe 1 is open."},
    ]

    state = PathPrivilegedContextProvider._execution_state_signature(
        events,
        "You open the safe 1. The safe 1 is open.",
        set(),
    )

    assert state["location"] == "drawer 6"


def test_state_recovery_correction_only_falls_back_when_candidate_matches_student():
    observation = "You arrive at desk 1. On the desk 1, you see a tissuebox 1."
    events = [
        {"action": "go to bed 1", "next_observation": "You arrive at bed 1."},
        {"action": "take pillow 1 from bed 1", "next_observation": "You pick up the pillow 1."},
        {"action": "go to desk 1", "next_observation": observation},
    ]
    trace_state = WalkthroughState(
        step_index=3,
        observation=observation,
        next_observation="You arrive at dresser 1.",
        admissible_actions=["go to dresser 1"],
        next_action="go to dresser 1",
        action_history=["go to bed 1", "take pillow 1 from bed 1", "go to desk 1"],
        observation_norm=observation.lower(),
        next_action_norm="go to dresser 1",
    )
    provider = _state_provider(events, trace_state)
    provider.recovery_correction_only = True
    provider.trace_cache = _FailIfCalledTraceCache()
    batch = SimpleNamespace(
        non_tensor_batch={
            "executed_action": np.array(["go to dresser 1"], dtype=object),
            "gamefile": np.array(["game.tw-pddl"], dtype=object),
            "turn_step": np.array([3], dtype=object),
        }
    )
    full_actions = ["go to bed 1", "take pillow 1 from bed 1", "go to desk 1", "go to dresser 1", "use desklamp 1"]

    recovery = provider._recovery_guess(
        batch=batch,
        idx=0,
        prompt_text=_prompt(observation, ["go to dresser 1", "go to bed 1"]),
        full_actions=full_actions,
        consumed=0,
    )

    assert recovery["route"] == "walkthrough_state_candidate_matches_student"
    assert recovery["action"] == ""
    assert not recovery["use_action"]
    assert recovery["state_match"]
    assert recovery["candidate_matches_student"]


def test_success_confirmation_weight_keeps_failure_trajectory_confirmation():
    provider = object.__new__(PathPrivilegedContextProvider)
    provider.recovery_success_confirmation_sdl_weight = 0.0
    batch = SimpleNamespace(
        non_tensor_batch={
            "executed_action": np.array(["go to dresser 1", "go to dresser 1", "go to desk 1"], dtype=object),
            "episode_rewards": np.array([10.0, 0.0, 10.0], dtype=object),
        }
    )

    success_match_weight, success_match_metrics = provider._apply_success_confirmation_sdl_weight(
        batch=batch, idx=0, candidate_action="go to dresser 1", sample_weight=1.0
    )
    failure_match_weight, failure_match_metrics = provider._apply_success_confirmation_sdl_weight(
        batch=batch, idx=1, candidate_action="go to dresser 1", sample_weight=1.0
    )
    success_mismatch_weight, success_mismatch_metrics = provider._apply_success_confirmation_sdl_weight(
        batch=batch, idx=2, candidate_action="go to dresser 1", sample_weight=1.0
    )

    assert success_match_weight == 0.0
    assert success_match_metrics["path_opd/recovery_success_confirmation_ratio"] == 1.0
    assert failure_match_weight == 1.0
    assert failure_match_metrics["path_opd/recovery_candidate_matches_sampled_ratio"] == 1.0
    assert failure_match_metrics["path_opd/recovery_success_confirmation_ratio"] == 0.0
    assert success_mismatch_weight == 1.0
    assert success_mismatch_metrics["path_opd/recovery_trajectory_success_ratio"] == 1.0
    assert success_mismatch_metrics["path_opd/recovery_candidate_matches_sampled_ratio"] == 0.0


def test_state_recovery_rejects_observation_match_with_inventory_mismatch():
    observation = "You arrive at dresser 1. On the dresser 1, you see a spraybottle 2."
    events = [
        {"action": "go to countertop 1", "next_observation": "You arrive at countertop 1."},
        {"action": "take toiletpaper 2 from countertop 1", "next_observation": "You pick up the toiletpaper 2."},
        {"action": "go to dresser 1", "next_observation": observation},
    ]
    trace_state = WalkthroughState(
        step_index=1,
        observation=observation,
        next_observation="You arrive at countertop 1.",
        admissible_actions=["go to countertop 1"],
        next_action="go to countertop 1",
        action_history=["go to dresser 1"],
        observation_norm=observation.lower(),
        next_action_norm="go to countertop 1",
    )
    provider = _state_provider(events, trace_state)
    batch = SimpleNamespace(
        non_tensor_batch={
            "executed_action": np.array(["go to drawer 2"], dtype=object),
            "gamefile": np.array(["game.tw-pddl"], dtype=object),
            "turn_step": np.array([3], dtype=object),
        }
    )
    full_actions = ["go to dresser 1", "go to countertop 1", "take toiletpaper 2 from countertop 1"]

    recovery = provider._recovery_guess(
        batch=batch,
        idx=0,
        prompt_text=_prompt(observation, ["go to countertop 1", "go to toiletpaperhanger 1"]),
        full_actions=full_actions,
        consumed=0,
    )

    assert recovery["route"] == "walkthrough_state_mismatch"
    assert recovery["action"] == ""
    assert not recovery["use_action"]
    assert not recovery["inventory_match"]


def test_state_recovery_allows_additional_monotonic_progress():
    observation = "You pick up the book 1 from the dresser 1."
    events = [
        {"action": "go to dresser 1", "next_observation": "You arrive at dresser 1."},
        {"action": "take book 2 from dresser 1", "next_observation": "You pick up the book 2."},
        {"action": "go to sofa 1", "next_observation": "You arrive at sofa 1."},
        {"action": "move book 2 to sofa 1", "next_observation": "You put the book 2 on the sofa 1."},
        {"action": "go to dresser 1", "next_observation": "You arrive at dresser 1."},
        {"action": "take book 1 from dresser 1", "next_observation": observation},
    ]
    trace_state = WalkthroughState(
        step_index=2,
        observation=observation,
        next_observation="You arrive at sofa 1.",
        admissible_actions=["go to sofa 1"],
        next_action="go to sofa 1",
        action_history=["go to dresser 1", "take book 1 from dresser 1"],
        observation_norm=observation.lower(),
        next_action_norm="go to sofa 1",
    )
    provider = _state_provider(events, trace_state)
    batch = SimpleNamespace(
        non_tensor_batch={
            "executed_action": np.array(["take book 1 from dresser 1"], dtype=object),
            "gamefile": np.array(["game.tw-pddl"], dtype=object),
            "turn_step": np.array([6], dtype=object),
        }
    )
    full_actions = [
        "go to dresser 1",
        "take book 1 from dresser 1",
        "go to sofa 1",
        "move book 1 to sofa 1",
        "go to dresser 1",
        "take book 2 from dresser 1",
        "go to sofa 1",
        "move book 2 to sofa 1",
    ]

    recovery = provider._recovery_guess(
        batch=batch,
        idx=0,
        prompt_text=_prompt(observation, ["go to sofa 1"]),
        full_actions=full_actions,
        consumed=0,
    )

    assert recovery["route"] == "walkthrough_state_compatible"
    assert recovery["action"] == "go to sofa 1"


def test_semantic_instance_match_grounds_different_pickup_object_id():
    observation = "You arrive at sofa 1 while carrying book 3."
    events = [
        {"action": "go to dresser 1", "next_observation": "You arrive at dresser 1."},
        {"action": "take book 3 from dresser 1", "next_observation": "You pick up the book 3."},
        {"action": "go to sofa 1", "next_observation": observation},
    ]
    provider = _state_provider(events, None)
    provider.recovery_alfworld_object_match_mode = "semantic_instance"
    batch = SimpleNamespace(
        non_tensor_batch={
            "executed_action": np.array(["go to sofa 1"], dtype=object),
            "gamefile": np.array(["game.tw-pddl"], dtype=object),
            "turn_step": np.array([3], dtype=object),
        }
    )
    full_actions = [
        "go to dresser 1",
        "take book 1 from dresser 1",
        "go to sofa 1",
        "move book 1 to sofa 1",
    ]

    recovery = provider._recovery_guess(
        batch=batch,
        idx=0,
        prompt_text=_prompt(observation, ["move book 3 to sofa 1", "go to dresser 1"]),
        full_actions=full_actions,
        consumed=0,
    )

    assert recovery["route"] == "walkthrough_state_compatible"
    assert recovery["action"] == "move book 3 to sofa 1"
    assert recovery["reference_action"] == "move book 1 to sofa 1"
    assert recovery["semantic_action_grounded"]
    assert recovery["grounded_action_count"] == 1
    assert recovery["state_signature"]["inventory"] == ("book 3",)
    assert recovery["trace_state_signature"]["inventory"] == ("book 1",)


def test_semantic_instance_match_grounds_equivalent_goal_receptacle_id():
    observation = "You arrive at shelf 1 while carrying book 3."
    events = [
        {"action": "go to dresser 1", "next_observation": "You arrive at dresser 1."},
        {"action": "take book 3 from dresser 1", "next_observation": "You pick up the book 3."},
        {"action": "go to shelf 1", "next_observation": observation},
    ]
    provider = _state_provider(events, None)
    provider.recovery_alfworld_object_match_mode = "semantic_instance"
    batch = SimpleNamespace(
        non_tensor_batch={
            "executed_action": np.array(["go to shelf 1"], dtype=object),
            "gamefile": np.array(["game.tw-pddl"], dtype=object),
            "turn_step": np.array([3], dtype=object),
        }
    )

    recovery = provider._recovery_guess(
        batch=batch,
        idx=0,
        prompt_text=_prompt(observation, ["move book 3 to shelf 1", "go to dresser 1"]),
        full_actions=[
            "go to dresser 1",
            "take book 1 from dresser 1",
            "go to shelf 2",
            "move book 1 to shelf 2",
        ],
        consumed=0,
    )

    assert recovery["route"] == "walkthrough_state_compatible"
    assert recovery["action"] == "move book 3 to shelf 1"
    assert recovery["reference_action"] == "move book 1 to shelf 2"
    assert recovery["semantic_action_grounded"]


def test_semantic_instance_match_grounds_equivalent_look_task_tool_id():
    observation = "You arrive at desk 2 while carrying pillow 2."
    events = [
        {"action": "go to bed 1", "next_observation": "You arrive at bed 1."},
        {"action": "take pillow 2 from bed 1", "next_observation": "You pick up the pillow 2."},
        {"action": "go to desk 2", "next_observation": observation},
    ]
    provider = _state_provider(events, None)
    provider.recovery_alfworld_object_match_mode = "semantic_instance"
    batch = SimpleNamespace(
        non_tensor_batch={
            "executed_action": np.array(["go to desk 2"], dtype=object),
            "gamefile": np.array(["game.tw-pddl"], dtype=object),
            "turn_step": np.array([3], dtype=object),
        }
    )

    recovery = provider._recovery_guess(
        batch=batch,
        idx=0,
        prompt_text=_prompt(observation, ["use desklamp 2", "go to bed 1"]),
        full_actions=[
            "go to bed 1",
            "take pillow 1 from bed 1",
            "go to desk 1",
            "use desklamp 1",
        ],
        consumed=0,
    )

    assert recovery["route"] == "walkthrough_state_compatible"
    assert recovery["action"] == "use desklamp 2"
    assert recovery["reference_action"] == "use desklamp 1"
    assert recovery["semantic_action_grounded"]


def test_exact_instance_match_still_rejects_different_pickup_object_id():
    observation = "You arrive at sofa 1 while carrying book 3."
    events = [
        {"action": "go to dresser 1", "next_observation": "You arrive at dresser 1."},
        {"action": "take book 3 from dresser 1", "next_observation": "You pick up the book 3."},
        {"action": "go to sofa 1", "next_observation": observation},
    ]
    provider = _state_provider(events, None)
    batch = SimpleNamespace(
        non_tensor_batch={
            "executed_action": np.array(["go to sofa 1"], dtype=object),
            "gamefile": np.array(["game.tw-pddl"], dtype=object),
            "turn_step": np.array([3], dtype=object),
        }
    )

    recovery = provider._recovery_guess(
        batch=batch,
        idx=0,
        prompt_text=_prompt(observation, ["move book 3 to sofa 1", "go to dresser 1"]),
        full_actions=[
            "go to dresser 1",
            "take book 1 from dresser 1",
            "go to sofa 1",
            "move book 1 to sofa 1",
        ],
        consumed=0,
    )

    assert recovery["route"] == "walkthrough_state_mismatch"
    assert not recovery["use_action"]


def test_semantic_instance_state_preserves_type_counts_and_properties():
    current = {
        "location": "sinkbasin 1",
        "inventory": ("mug 3",),
        "object_locations": (("mug 2", "countertop 1"),),
        "object_properties": ("clean:mug 3",),
        "activated_tools": (),
    }
    compatible_reference = {
        "location": "sinkbasin 1",
        "inventory": ("mug 1",),
        "object_locations": (("mug 4", "countertop 1"),),
        "object_properties": ("clean:mug 1",),
        "activated_tools": (),
    }
    too_much_progress_reference = {
        **compatible_reference,
        "object_locations": (("mug 1", "countertop 1"), ("mug 4", "countertop 1")),
    }
    wrong_type_reference = {**compatible_reference, "inventory": ("plate 1",)}

    assert all(
        PathPrivilegedContextProvider._state_component_compatibility(
            current,
            compatible_reference,
            "semantic_instance",
        ).values()
    )
    assert not PathPrivilegedContextProvider._state_component_compatibility(
        current,
        too_much_progress_reference,
        "semantic_instance",
    )["object_locations"]
    assert not PathPrivilegedContextProvider._state_component_compatibility(
        current,
        wrong_type_reference,
        "semantic_instance",
    )["inventory"]


def test_strict_alignment_does_not_expose_non_admissible_buffer_action(tmp_path):
    index_path = tmp_path / "paths.jsonl"
    index_path.write_text(
        json.dumps(
            {
                "gamefile": "game.tw-pddl",
                "full_actions": ["go to diningtable 1", "take potato 1 from diningtable 1", "go to countertop 1"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = OmegaConf.create(
        {
            "algorithm": {
                "path_opd": {
                    "teacher_context": "gt_recovery_aligned",
                    "trajectory_index_path": str(index_path),
                    "alignment_mode": "dynamic",
                    "path_source_policy": "canonical",
                    "enable_walkthrough_trace_recovery": False,
                }
            }
        }
    )
    provider = PathPrivilegedContextProvider(config=config)
    first_prompt = _prompt("You are in the middle of a room.", ["go to diningtable 1"])
    current_prompt = _prompt(
        "You arrive at diningtable 1. On the diningtable 1, you see a bread 1.",
        ["go to countertop 1", "take bread 1 from diningtable 1"],
    )
    batch = SimpleNamespace(
        non_tensor_batch={
            "traj_uid": np.array(["traj-1", "traj-1"], dtype=object),
            "turn_step": np.array([0, 1], dtype=object),
            "executed_action": np.array(["go to diningtable 1", "go to countertop 1"], dtype=object),
            "is_action_valid": np.array([True, True], dtype=object),
            "input": np.array([first_prompt, current_prompt], dtype=object),
            "gamefile": np.array(["game.tw-pddl", "game.tw-pddl"], dtype=object),
        }
    )

    provider.start_batch_metrics()
    context = provider.build_context(batch=batch, idx=1, prompt_text=current_prompt)
    metrics = provider._sample_metrics[-1]

    assert "Candidate next action" not in context
    assert metrics["path_opd/recovery_route_strict_next_action_non_admissible_ratio"] == 1.0


def test_strict_alignment_fails_closed_without_parseable_admissible_actions(tmp_path):
    index_path = tmp_path / "paths.jsonl"
    index_path.write_text(
        json.dumps({"gamefile": "game.tw-pddl", "full_actions": ["go to diningtable 1"]}) + "\n",
        encoding="utf-8",
    )
    config = OmegaConf.create(
        {
            "algorithm": {
                "path_opd": {
                    "teacher_context": "gt_recovery_aligned",
                    "trajectory_index_path": str(index_path),
                    "alignment_mode": "dynamic",
                    "path_source_policy": "canonical",
                    "enable_walkthrough_trace_recovery": False,
                }
            }
        }
    )
    provider = PathPrivilegedContextProvider(config=config)
    prompt = "Your task is to: examine the pillow with the desklamp.\nNo action list is available."
    batch = SimpleNamespace(
        non_tensor_batch={
            "traj_uid": np.array(["traj-1"], dtype=object),
            "turn_step": np.array([0], dtype=object),
            "executed_action": np.array([""], dtype=object),
            "is_action_valid": np.array([True], dtype=object),
            "input": np.array([prompt], dtype=object),
            "gamefile": np.array(["game.tw-pddl"], dtype=object),
        }
    )

    provider.start_batch_metrics()
    context = provider.build_context(batch=batch, idx=0, prompt_text=prompt)
    metrics = provider._sample_metrics[-1]

    assert "Candidate next action" not in context
    assert metrics["path_opd/recovery_route_strict_next_action_non_admissible_ratio"] == 1.0
