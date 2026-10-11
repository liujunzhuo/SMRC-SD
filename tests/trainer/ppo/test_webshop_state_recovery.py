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

import json
import numpy as np
from omegaconf import OmegaConf

from agent_system.webshop_state_utils import (
    build_webshop_oracle_manifest_record,
    build_webshop_oracle_reference,
    webshop_task_id,
    webshop_action_is_admissible,
    webshop_progress_state_key,
)
from agent_system.environments.env_package.webshop.envs import WebshopMultiProcessEnv
from verl.trainer.ppo.path_privileged_utils import (
    PathCandidate,
    PathPrivilegedContextProvider,
    PathTrajectoryIndex,
    build_state_sdl_selection_mask,
    extract_admissible_actions,
)


def _goal():
    return {
        "asin": "B012345678",
        "name": "Example Running Shoe, Lightweight Blue Trainer for Adults",
        "goal_options": {"color": "blue", "size": "large"},
        "instruction_text": "Find a lightweight blue running shoe in size large",
    }


def _prompt(observation, actions):
    available = "\n".join(f"'{action}'," for action in actions)
    return (
        "You are an expert autonomous agent operating in WebShop.\n"
        "Your task is to: Find a lightweight blue running shoe in size large.\n"
        f"Your current observation is: {observation}.\n"
        "Your admissible actions of the current situation are:\n"
        f"[\n{available}\n].\n\n"
        "Now it's your turn to take one action for the current step."
    )


def _provider(
    teacher_context="gt_recovery_aligned",
    webshop_match_mode="canonical_exact",
    *,
    state_match_context="recovery",
    state_sdl_selection="all",
):
    return PathPrivilegedContextProvider(
        config=OmegaConf.create(
            {
                "algorithm": {
                    "path_opd": {
                        "teacher_context": teacher_context,
                        "strict_gt_path": True,
                        "alignment_mode": "dynamic" if teacher_context == "gt_recovery_aligned" else "full_start",
                        "enable_walkthrough_trace_recovery": teacher_context == "gt_recovery_aligned",
                        "recovery_match_mode": "state" if teacher_context == "gt_recovery_aligned" else "history",
                        "recovery_state_domain": "webshop",
                        "recovery_webshop_match_mode": webshop_match_mode,
                        "state_match_context": state_match_context,
                        "state_sdl_selection": state_sdl_selection,
                        "trajectory_index_path": None,
                    }
                }
            }
        )
    )


def _batch(state, reference, prompt, executed_action="search[other query]"):
    metadata = {
        "full_actions": reference["full_actions"],
        "path_source": "webshop_goal_oracle",
        "webshop_reference_states": reference["reference_states"],
        "webshop_target_asin": reference["target_asin"],
        "webshop_target_options": reference["target_options"],
    }
    return SimpleNamespace(
        non_tensor_batch={
            "traj_uid": np.array(["traj-1"], dtype=object),
            "turn_step": np.array([1], dtype=object),
            "executed_action": np.array([executed_action], dtype=object),
            "is_action_valid": np.array([True], dtype=object),
            "input": np.array([prompt], dtype=object),
            "task_key": np.array(["500"], dtype=object),
            "gamefile": np.array([None], dtype=object),
            "webshop_state": np.array([state], dtype=object),
            "vmpr_metadata": np.array([metadata], dtype=object),
        }
    )


def test_webshop_oracle_reference_tracks_pre_action_progress_states():
    reference = build_webshop_oracle_reference(_goal(), query_variant="short_name")

    assert reference["full_actions"] == [
        "search[Example Running Shoe]",
        "click[B012345678]",
        "click[blue]",
        "click[large]",
        "click[buy now]",
    ]
    assert len(reference["reference_states"]) == len(reference["full_actions"])
    assert reference["reference_states"][0]["page_type"] == "search_home"
    assert reference["reference_states"][1]["page_type"] == "search_results"
    assert reference["reference_states"][2]["asin"] == "b012345678"
    assert reference["reference_states"][3]["options"] == {"color": "blue"}
    assert reference["reference_states"][4]["options"] == {"color": "blue", "size": "large"}


def test_webshop_task_id_is_stable_across_option_order_and_manifest_index():
    goal = _goal()
    reordered = {**goal, "goal_options": {"size": "large", "color": "blue"}}

    assert webshop_task_id(goal) == webshop_task_id(reordered)
    record = build_webshop_oracle_manifest_record(
        goal,
        goal_index=17,
        query_variant="full_name",
        expected_score=1.0,
    )
    assert record["goal_index"] == 17
    assert record["webshop_task_id"] == webshop_task_id(goal, goal_index=17)
    assert record["full_actions"][-1] == "click[buy now]"


def test_webshop_pooled_validation_maps_replicas_to_same_fixed_goal():
    env = WebshopMultiProcessEnv.__new__(WebshopMultiProcessEnv)
    env.is_train = False
    env.fixed_goal_indices = [41, 73]
    env.task_ids = [f"task-{index}" for index in range(100)]

    reset_kwargs = env.validation_reset_kwargs(["0:0", "0:0", "0:1", "0:1"])

    assert [item["session_id"] for item in reset_kwargs] == [41, 41, 73, 73]
    assert [item["expected_webshop_task_id"] for item in reset_kwargs] == [
        "task-41",
        "task-41",
        "task-73",
        "task-73",
    ]


def test_webshop_progress_state_ignores_query_and_page_but_not_product_options():
    lhs = {"page_type": "search_results", "keywords": ["shoe"], "page": 1, "asin": "", "options": {}}
    rhs = {"page_type": "search_results", "keywords": ["trainer"], "page": 3, "asin": "", "options": {}}
    wrong_product = {"page_type": "item_page", "asin": "other", "options": {}}
    target_product = {"page_type": "item_page", "asin": "B012345678", "options": {}}

    assert webshop_progress_state_key(lhs) == webshop_progress_state_key(rhs)
    assert webshop_progress_state_key(wrong_product) != webshop_progress_state_key(target_product)


def test_webshop_search_placeholder_admits_concrete_oracle_query():
    assert webshop_action_is_admissible("search[exact product name]", ["search[<your query>]", "click[search]"])
    assert not webshop_action_is_admissible("click[B012345678]", ["search[<your query>]", "click[search]"])


def test_webshop_admissible_parser_preserves_option_values_with_apostrophes():
    prompt = _prompt("Target product", ["click[lavender]", "click[20''x20'']"])

    assert extract_admissible_actions(prompt) == ["click[lavender]", "click[20''x20'']"]


def _alfworld_prompt(actions):
    """Reproduce ALFWorld's numpy-style admissible action repr.

    The first action shares the preamble line after ``[`` and the last action is
    followed by ``].``; items are space-separated with no commas.
    """
    body = "\n ".join(f"'{action}'" for action in actions)
    return (
        "You are an expert agent operating in the ALFRED Embodied Environment.\n"
        "Your task is to: put some alarmclock on desk.\n"
        f"Your admissible actions of the current situation are: [{body}].\n\n"
        "Now it's your turn to take an action."
    )


def test_alfworld_admissible_parser_keeps_first_and_last_actions():
    actions = [
        "examine bed 1",
        "go to desk 1",
        "go to dresser 1",
        "take pillow 1 from bed 1",
    ]
    prompt = _alfworld_prompt(actions)

    # Regression guard: the numpy repr must not drop the first ("examine ...")
    # nor the last ("take ... from ...") admissible action.
    assert extract_admissible_actions(prompt) == actions


def test_alfworld_admissible_parser_handles_multiple_actions_per_line():
    prompt = (
        "Your admissible actions of the current situation are: ['inventory' 'look'\n"
        " 'go to desk 1'].\n\n"
        "Now it's your turn to take an action."
    )

    assert extract_admissible_actions(prompt) == ["inventory", "look", "go to desk 1"]


def test_webshop_d0_reads_oracle_path_from_reset_metadata():
    reference = build_webshop_oracle_reference(_goal())
    prompt = _prompt("Search results", ["click[B012345678]"])
    batch = _batch(reference["reference_states"][1], reference, prompt)
    provider = _provider(teacher_context="gt_path")

    provider.start_batch_metrics()
    context = provider.build_context(batch=batch, idx=0, prompt_text=prompt)

    assert "Complete successful path for this task:" in context
    assert "search[Example Running Shoe]" in context
    assert "click[B012345678]" in context
    assert "Candidate next action" not in context


def test_webshop_d2rs_matches_progress_state_and_exposes_admissible_next_action():
    reference = build_webshop_oracle_reference(_goal())
    current_state = {
        **reference["reference_states"][1],
        "keywords": ["different", "query"],
        "page": 4,
    }
    prompt = _prompt("A different search result page", ["click[B012345678]", "click[other]"])
    batch = _batch(current_state, reference, prompt)
    provider = _provider()

    provider.start_batch_metrics()
    context = provider.build_context(batch=batch, idx=0, prompt_text=prompt)
    metrics = provider._sample_metrics[-1]

    assert "Your current shopping progress matches the state after these successful-path actions:" in context
    assert "search[Example Running Shoe]" in context
    assert "Current state summary:" in context
    assert "page_type=search_results" in context
    assert "Candidate next action for the current state:\nclick[B012345678]" in context
    assert metrics["path_opd/recovery_state_match_ratio"] == 1.0
    assert metrics["path_opd/recovery_webshop_page_type_match_ratio"] == 1.0
    assert metrics["path_opd/recovery_webshop_asin_match_ratio"] == 1.0
    assert metrics["path_opd/recovery_webshop_options_match_ratio"] == 1.0


def test_webshop_d2rs_wrong_product_falls_back_without_candidate_action():
    reference = build_webshop_oracle_reference(_goal())
    wrong_state = {"page_type": "item_page", "asin": "B999999999", "options": {}}
    prompt = _prompt("Wrong product page", ["click[blue]", "click[buy now]"])
    batch = _batch(wrong_state, reference, prompt, executed_action="click[other]")
    provider = _provider()

    provider.start_batch_metrics()
    context = provider.build_context(batch=batch, idx=0, prompt_text=prompt)
    metrics = provider._sample_metrics[-1]

    assert "Candidate next action for the current state:" not in context
    assert metrics["path_opd/recovery_state_match_ratio"] == 0.0
    assert metrics["path_opd/recovery_route_webshop_progress_state_mismatch_ratio"] == 1.0


def test_webshop_d0_matched_keeps_state_match_metric_but_emits_full_path_context():
    """D0@Matched is shared PathPrivilegedContextProvider logic, not ALFWorld-only.

    WebShop goal_progress matcher still records recovery_state_match_ratio for
    match_only row selection, while state_match_context=full_path suppresses the
    localized goal-progress / candidate prompt in favor of plain D0 full path.
    """
    reference = build_webshop_oracle_reference(_goal())
    state = {"page_type": "search_results", "asin": "", "options": {}}
    prompt = _prompt("Search results", ["click[B012345678]", "click[other]"])
    batch = _batch(state, reference, prompt)

    recovery_provider = _provider(webshop_match_mode="goal_progress")
    recovery_provider.start_batch_metrics()
    recovery_context = recovery_provider.build_context(batch=batch, idx=0, prompt_text=prompt)
    recovery_metrics = recovery_provider._sample_metrics[-1]
    assert "Complete successful path for this task:" in recovery_context
    assert "Current state summary:" in recovery_context
    assert "Candidate next action for the current state:\nclick[B012345678]" in recovery_context
    assert float(recovery_metrics.get("path_opd/recovery_state_match_ratio", 0.0)) == 1.0

    d0_provider = _provider(
        webshop_match_mode="goal_progress",
        state_match_context="full_path",
        state_sdl_selection="match_only",
    )
    d0_provider.start_batch_metrics()
    d0_context = d0_provider.build_context(batch=batch, idx=0, prompt_text=prompt)
    d0_metrics = d0_provider._sample_metrics[-1]

    assert "Complete successful path for this task:" in d0_context
    assert "The current state may be on or off this path." in d0_context
    assert "Candidate next action for the current state:" not in d0_context
    # D0@Matched suppresses localized summary/candidate even though the matcher matched.
    assert "Search results; reference target product" not in d0_context
    assert float(d0_metrics.get("path_opd/recovery_state_match_ratio", 0.0)) == 1.0

    selection_mask, selection_metrics = build_state_sdl_selection_mask(
        [d0_metrics, {"path_opd/recovery_state_match_ratio": 0.0}],
        mode="match_only",
    )
    assert selection_mask == [1.0, 0.0]
    assert selection_metrics["path_opd/state_sdl_selection_match_only"] == 1.0
    assert selection_metrics["path_opd/state_sdl_match_count"] == 1.0


def test_webshop_goal_progress_accepts_reversed_option_order():
    reference = build_webshop_oracle_reference(_goal())
    state = {
        "page_type": "item_page",
        "asin": "B012345678",
        "options": {"size": "large"},
    }
    prompt = _prompt("Target product with size already selected", ["click[blue]", "click[buy now]"])
    batch = _batch(state, reference, prompt, executed_action="click[blue]")
    provider = _provider(webshop_match_mode="goal_progress")

    provider.start_batch_metrics()
    context = provider.build_context(batch=batch, idx=0, prompt_text=prompt)
    metrics = provider._sample_metrics[-1]

    assert "Complete successful path for this task:" in context
    assert "Current state summary:" in context
    assert "selected=size=large" in context
    assert "remaining(any order)=color=blue" in context
    assert "Candidate next action for the current state:\nclick[blue]" in context
    assert metrics["path_opd/recovery_state_match_ratio"] == 1.0
    assert metrics["path_opd/recovery_webshop_unique_candidate_ratio"] == 1.0


def test_webshop_goal_progress_uses_set_guidance_when_option_order_is_not_unique():
    reference = build_webshop_oracle_reference(_goal())
    state = {"page_type": "item_page", "asin": "B012345678", "options": {}}
    prompt = _prompt("Target product page", ["click[blue]", "click[large]", "click[buy now]"])
    batch = _batch(state, reference, prompt, executed_action="click[large]")
    provider = _provider(webshop_match_mode="goal_progress")

    provider.start_batch_metrics()
    context = provider.build_context(batch=batch, idx=0, prompt_text=prompt)
    metrics = provider._sample_metrics[-1]

    assert "Complete successful path for this task:" in context
    assert "remaining(any order)=color=blue, size=large" in context
    assert "Compatible next actions (choose one based on the current observation):" in context
    assert "click[blue], click[large]" in context
    assert "Candidate next action for the current state:" not in context
    assert "Do not buy until all required options are selected." not in context
    assert metrics["path_opd/recovery_state_match_ratio"] == 1.0
    assert metrics["path_opd/recovery_webshop_set_guidance_ratio"] == 1.0
    assert metrics["path_opd/recovery_webshop_candidate_set_size_mean"] == 2.0


def test_webshop_goal_progress_does_not_force_one_search_query():
    reference = build_webshop_oracle_reference(_goal())
    state = {"page_type": "search_home", "asin": "", "options": {}}
    prompt = _prompt("Search", ["search[<your query>]", "click[search]"])
    batch = _batch(state, reference, prompt, executed_action="search[blue large running shoe]")
    provider = _provider(webshop_match_mode="goal_progress")

    provider.start_batch_metrics()
    context = provider.build_context(batch=batch, idx=0, prompt_text=prompt)
    metrics = provider._sample_metrics[-1]

    assert "Complete successful path for this task:" in context
    assert "One successful reference query:" in context
    assert "Candidate next action for the current state:" not in context
    assert metrics["path_opd/recovery_state_match_ratio"] == 1.0
    assert metrics["path_opd/recovery_webshop_set_guidance_ratio"] == 1.0


def test_webshop_goal_progress_exposes_target_only_when_visible():
    reference = build_webshop_oracle_reference(_goal())
    state = {"page_type": "search_results", "asin": "", "options": {}}
    visible_prompt = _prompt("Search results", ["click[B012345678]", "click[other]"])
    hidden_prompt = _prompt("Different search results", ["click[other]"])
    provider = _provider(webshop_match_mode="goal_progress")

    provider.start_batch_metrics()
    visible_context = provider.build_context(
        batch=_batch(state, reference, visible_prompt), idx=0, prompt_text=visible_prompt
    )
    hidden_context = provider.build_context(
        batch=_batch(state, reference, hidden_prompt), idx=0, prompt_text=hidden_prompt
    )

    assert "Candidate next action for the current state:\nclick[B012345678]" in visible_context
    assert "Candidate next action for the current state:" not in hidden_context
    assert "Complete successful path for this task:" in hidden_context


def test_webshop_goal_progress_rejects_wrong_selected_option_and_buys_only_when_ready():
    reference = build_webshop_oracle_reference(_goal())
    wrong_state = {
        "page_type": "item_page",
        "asin": "B012345678",
        "options": {"color": "red"},
    }
    ready_state = {
        "page_type": "item_page",
        "asin": "B012345678",
        "options": {"size": "large", "color": "blue"},
    }
    wrong_prompt = _prompt("Wrong color selected", ["click[blue]", "click[buy now]"])
    ready_prompt = _prompt("All required options selected", ["click[buy now]"])
    provider = _provider(webshop_match_mode="goal_progress")

    provider.start_batch_metrics()
    wrong_context = provider.build_context(
        batch=_batch(wrong_state, reference, wrong_prompt), idx=0, prompt_text=wrong_prompt
    )
    ready_context = provider.build_context(
        batch=_batch(ready_state, reference, ready_prompt), idx=0, prompt_text=ready_prompt
    )

    assert "Candidate next action for the current state:" not in wrong_context
    assert "Complete successful path for this task:" in wrong_context
    assert "ready_to_buy" in ready_context
    assert "Candidate next action for the current state:\nclick[buy now]" in ready_context


def test_webshop_goal_progress_wrong_asin_stays_unmatched():
    """Wrong-product item pages remain unmatched under MatchOnly (no prev route)."""
    reference = build_webshop_oracle_reference(_goal())
    wrong_state = {"page_type": "item_page", "asin": "B999999999", "options": {"color": "red"}}
    prompt = _prompt(
        "Wrong product page",
        ["click[back to search]", "click[< prev]", "click[red]", "click[buy now]"],
    )
    provider = _provider(webshop_match_mode="goal_progress")

    provider.start_batch_metrics()
    context = provider.build_context(batch=_batch(wrong_state, reference, prompt), idx=0, prompt_text=prompt)
    metrics = provider._sample_metrics[-1]

    assert "Complete successful path for this task:" in context
    assert "Candidate next action for the current state:" not in context
    assert "click[< prev]" not in context
    assert float(metrics.get("path_opd/recovery_state_match_ratio", 0.0)) == 0.0
    assert float(metrics.get("path_opd/recovery_route_webshop_goal_progress_incompatible_ratio", 0.0)) == 1.0


def test_path_trajectory_index_keys_webshop_task_id_from_oracle_manifest(tmp_path):
    goal = _goal()
    record = build_webshop_oracle_manifest_record(goal, goal_index=7, query_variant="short_name")
    path = tmp_path / "oracle.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    index = PathTrajectoryIndex.from_jsonl(str(path))

    assert index.get(record["webshop_task_id"]) == record["full_actions"]
    assert index.get("7") == record["full_actions"]


def test_best_state_aligned_prefers_webshop_oracle_when_quality_is_tied():
    reference = build_webshop_oracle_reference(_goal())
    state = {
        "page_type": "item_page",
        "asin": "B012345678",
        "options": {"size": "large", "color": "blue"},
    }
    prompt = _prompt("Ready to buy", ["click[buy now]"])
    batch = _batch(state, reference, prompt, executed_action="click[buy now]")
    provider = _provider(webshop_match_mode="goal_progress")
    provider.path_source_policy = "best_state_aligned_success"

    selected, recovery, metrics = provider._select_state_compatible_path_candidate(
        batch=batch,
        idx=0,
        candidates=[
            PathCandidate(actions=list(reference["full_actions"]), source="webshop_goal_oracle"),
            PathCandidate(
                actions=["click[B012345678]", "click[blue]", "click[large]", "click[buy now]"],
                source="student_autonomous",
                insert_step=99,
            ),
        ],
        prompt_text=prompt,
    )

    assert selected.source == "webshop_goal_oracle"
    assert recovery is not None and recovery.get("use_action")
    assert metrics["path_opd/path_has_state_compatible_buffer_candidate_ratio"] == 1.0
    assert metrics["path_opd/path_selected_state_compatible_buffer_candidate_ratio"] == 0.0


def test_best_state_aligned_uses_student_when_only_student_is_compatible():
    reference = build_webshop_oracle_reference(_goal())
    state = {
        "page_type": "item_page",
        "asin": "B012345678",
        "options": {"size": "large", "color": "blue"},
    }
    prompt = _prompt("Ready to buy", ["click[buy now]"])
    batch = _batch(state, reference, prompt, executed_action="click[buy now]")
    provider = _provider(webshop_match_mode="goal_progress")
    provider.path_source_policy = "best_state_aligned_success"

    selected, recovery, metrics = provider._select_state_compatible_path_candidate(
        batch=batch,
        idx=0,
        candidates=[
            # Empty canonical cannot enter the compatible set.
            PathCandidate(actions=[], source="webshop_goal_oracle"),
            PathCandidate(
                actions=["click[B012345678]", "click[blue]", "click[large]", "click[buy now]"],
                source="student_autonomous",
            ),
        ],
        prompt_text=prompt,
    )

    assert selected.source == "student_autonomous"
    assert recovery is not None and recovery.get("use_action")
    assert metrics["path_opd/path_selected_state_compatible_buffer_candidate_ratio"] == 1.0
