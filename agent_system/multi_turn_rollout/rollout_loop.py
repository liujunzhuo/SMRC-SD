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

import math
import random
import re
from collections import Counter

import torch
import numpy as np
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
import uuid
from agent_system.multi_turn_rollout.utils import process_image, to_list_of_dict, torch_to_numpy, filter_group_data
from agent_system.environments import EnvironmentManagerBase
from agent_system.multi_turn_rollout.prefix_buffer import ASSISTED_SOURCE, FULL_START_SOURCE, STUDENT_SOURCE, TEACHER_SOURCE, PrefixBuffer
from typing import Any, Dict, List, Optional, Tuple
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from agent_system.webshop_state_utils import webshop_action_is_admissible
from verl.trainer.ppo.path_privileged_utils import (
    PathTrajectoryIndex,
    compact_action_text,
    extract_admissible_actions,
    extract_current_observation,
    extract_task_text,
    infer_location_from_observation,
    normalize_action_text,
    normalize_observation_text,
)

class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        """
        Initialize the TrajectoryProcessor class.
        
        Parameters:
            config: Configuration object containing data processing settings
            tokenizer (PreTrainedTokenizer): Tokenizer for text encoding and decoding
            processor: Image processor for multimodal inputs
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.vmpr_cfg = self.config.algorithm.get("vmpr", {}) if hasattr(self.config, "algorithm") else {}
        self.vmpr_enabled = bool(self.vmpr_cfg.get("enable", False))
        self.vmpr_seed = int(self.vmpr_cfg.get("seed", 0))
        self.vmpr_rng = random.Random(self.vmpr_seed)
        self.prefix_buffer = None
        self._vmpr_pending_metrics = {}
        path_opd_cfg = self.config.algorithm.get("path_opd", {}) if hasattr(self.config, "algorithm") else {}
        self.vmpr_success_max_action_len = int(self.vmpr_cfg.get("success_max_action_len", 0) or 0)
        self.vmpr_success_max_length_ratio_to_canonical = float(self.vmpr_cfg.get("success_max_length_ratio_to_canonical", 0.0) or 0.0)
        self.vmpr_success_trajectory_index_path = self.vmpr_cfg.get("success_trajectory_index_path") or path_opd_cfg.get("trajectory_index_path")
        self._vmpr_success_trajectory_index: Optional[PathTrajectoryIndex] = None
        self.path_opd_student_context = str(path_opd_cfg.get("student_context", "none")).lower()
        if self.path_opd_student_context not in {"none", "state_history"}:
            raise ValueError(f"Unsupported algorithm.path_opd.student_context={self.path_opd_student_context}. Expected none or state_history.")
        alfworld_cfg = self.config.env.get("alfworld", {}) if hasattr(self.config, "env") else {}
        self.penalize_extra_output_after_action = bool(alfworld_cfg.get("penalize_extra_output_after_action", False))
        if self.vmpr_enabled:
            self.prefix_buffer = PrefixBuffer(
                env_name=self.config.env.env_name,
                seed_prefix_path=self.vmpr_cfg.get("seed_prefix_path"),
                buffer_path=self.vmpr_cfg.get("buffer_path"),
                max_buffer_size=self.vmpr_cfg.get("max_buffer_size", 4096),
                step_boundaries=self.vmpr_cfg.get("step_boundaries", []),
                teacher_source_weight=self.vmpr_cfg.get("teacher_source_weight", 1.0),
                student_source_weight=self.vmpr_cfg.get("student_source_weight", 1.0),
                assisted_source_weight=self.vmpr_cfg.get("assisted_source_weight", 0.2),
                max_teacher_buffer_size=self.vmpr_cfg.get("max_teacher_buffer_size", None),
                max_student_buffer_size=self.vmpr_cfg.get("max_student_buffer_size", None),
                max_assisted_buffer_size=self.vmpr_cfg.get("max_assisted_buffer_size", None),
                preserve_teacher_seed=self.vmpr_cfg.get("preserve_teacher_seed", False),
                source_sampling_strategy=self.vmpr_cfg.get("source_sampling_strategy", "entry_weighted"),
                max_prefix_use_count=self.vmpr_cfg.get("max_prefix_use_count", 0),
                max_student_prefix_use_count=self.vmpr_cfg.get("max_student_prefix_use_count", 0),
                drop_prefix_after_sample=self.vmpr_cfg.get("drop_prefix_after_sample", False),
                success_ema_sampling_weight=self.vmpr_cfg.get("success_ema_sampling_weight", 0.0),
                freshness_decay=self.vmpr_cfg.get("freshness_decay", 0.001),
                student_insertion_freshness_decay=self.vmpr_cfg.get("student_insertion_freshness_decay", 0.0),
                student_prefix_boundary_sample=self.vmpr_cfg.get("student_prefix_boundary_sample", "all"),
                max_student_prefix_length_ratio=self.vmpr_cfg.get("max_student_prefix_length_ratio", 0.0),
                seed=self.vmpr_seed + 1,
            )

    @staticmethod
    def _path_opd_inventory_from_actions(actions: List[str]) -> List[str]:
        carried: set[str] = set()
        for action in actions:
            text = normalize_action_text(action)
            take_match = re.match(r"take (.+?) from .+", text)
            if take_match:
                carried.add(take_match.group(1))
                continue
            move_match = re.match(r"(?:move|put) (.+?) (?:to|in|on) .+", text)
            if move_match:
                carried.discard(move_match.group(1))
                continue
            drop_match = re.match(r"drop (.+)", text)
            if drop_match:
                carried.discard(drop_match.group(1))
        return sorted(carried)

    @staticmethod
    def _path_opd_visited_locations(actions: List[str]) -> List[str]:
        visited: List[str] = []
        seen: set[str] = set()
        for action in actions:
            match = re.match(r"go to (.+)", normalize_action_text(action))
            if not match:
                continue
            location = match.group(1).strip()
            if location and location not in seen:
                visited.append(location)
                seen.add(location)
        return visited

    @staticmethod
    def _path_opd_repeated_actions(actions: List[str]) -> List[str]:
        counts = Counter(normalize_action_text(action) for action in actions if action)
        return [action for action, count in counts.most_common() if action and count > 1][:8]

    @staticmethod
    def _path_opd_insert_context(obs_content: str, context_text: str) -> str:
        if not context_text:
            return obs_content
        if context_text.lstrip().startswith("[History Summary]"):
            history_text = context_text
            remaining_context = ""
            end_marker = "[/History Summary]"
            end_idx = context_text.find(end_marker)
            if end_idx >= 0:
                split_idx = end_idx + len(end_marker)
                history_text = context_text[:split_idx].strip()
                remaining_context = context_text[split_idx:].strip()
            prior_match = re.search(r"(Prior to this step, you have already taken \d+ step\(s\)\.)", obs_content)
            if prior_match:
                insert_at = prior_match.end()
                obs_content = f"{obs_content[:insert_at]}\n\n{history_text}\n\n{obs_content[insert_at:].lstrip()}"
            else:
                marker = "\nYou are now at step "
                if marker in obs_content:
                    obs_content = obs_content.replace(marker, f"\n\n{history_text}\n{marker}", 1)
                else:
                    fallback_marker = "\nYour admissible actions of the current situation are:"
                    if fallback_marker in obs_content:
                        obs_content = obs_content.replace(fallback_marker, f"\n\n{history_text}\n{fallback_marker}", 1)
                    else:
                        obs_content = f"{obs_content}\n\n{history_text}"
            if remaining_context:
                marker = "\n\nNow it's your turn to take an action."
                if marker in obs_content:
                    obs_content = obs_content.replace(marker, f"\n\n{remaining_context}\n" + marker, 1)
                else:
                    obs_content = f"{obs_content}\n\n{remaining_context}"
            return TrajectoryCollector._path_opd_make_history_reasoning_instruction(obs_content)

        marker = "\n\nNow it's your turn to take an action."
        if marker in obs_content:
            return obs_content.replace(marker, f"\n\n{context_text}\n" + marker, 1)
        return f"{obs_content}\n\n{context_text}"

    @staticmethod
    def _path_opd_make_history_reasoning_instruction(obs_content: str) -> str:
        no_think_old = (
            "Now it's your turn to take an action.\n"
            "You should first reason step-by-step about the current situation.\n"
            "Once you've finished your reasoning, you should choose exactly one admissible action for current step and present it within <action> </action> tags."
        )
        no_think_new = (
            "Now it's your turn to take an action.\n"
            "You should first reason step-by-step about the current situation using the current observation, the History Summary, and the admissible actions.\n"
            "Use the History Summary to track what has already been tried and avoid unproductive repeats.\n"
            "Once you've finished your reasoning, you should choose exactly one admissible action for current step and present it within <action> </action> tags."
        )
        if no_think_old in obs_content:
            return obs_content.replace(no_think_old, no_think_new, 1)

        think_old = (
            "Now it's your turn to take an action.\n"
            "You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. \n"
            "Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags."
        )
        think_new = (
            "Now it's your turn to take an action.\n"
            "You should first reason step-by-step about the current situation using the current observation, the History Summary, and the admissible actions. This reasoning process MUST be enclosed within <think> </think> tags. \n"
            "Use the History Summary to track what has already been tried and avoid unproductive repeats.\n"
            "Once you've finished your reasoning, you should choose exactly one admissible action for current step and present it within <action> </action> tags."
        )
        if think_old in obs_content:
            return obs_content.replace(think_old, think_new, 1)

        action_only_old = (
            "Now it's your turn to take an action.\n"
            "Choose exactly one admissible action for current step and present it within <action> </action> tags."
        )
        action_only_new = (
            "Now it's your turn to take an action.\n"
            "Briefly use the current observation, the History Summary, and the admissible actions to avoid unproductive repeats.\n"
            "Choose exactly one admissible action for current step and present it within <action> </action> tags."
        )
        if action_only_old in obs_content:
            return obs_content.replace(action_only_old, action_only_new, 1)
        return obs_content

    @staticmethod
    def _has_extra_output_after_first_action(response_text: str) -> bool:
        """Return whether the model continues generating after the first action block."""
        if not response_text:
            return False
        lower_text = response_text.lower()
        end_tag = "</action>"
        end_idx = lower_text.find(end_tag)
        if end_idx < 0:
            return False
        tail = response_text[end_idx + len(end_tag):]
        tail = re.sub(r"(<\|im_end\|>|<\|endoftext\|>|<\|eot_id\|>|</s>)", "", tail, flags=re.IGNORECASE)
        return bool(tail.strip())

    def _path_opd_state_block_from_rollout_history(self, item: int, obs_content: str, state_history: Dict[str, Any]) -> str:
        current_observation = extract_current_observation(obs_content) or obs_content
        current_location = infer_location_from_observation(current_observation)
        if current_location == "unknown":
            current_location = str(state_history.get("last_location") or "unknown")
        valid_actions = [str(action) for action in state_history.get("valid_actions", []) if action]
        last_invalid_action = str(state_history.get("last_invalid_action") or "").strip()
        recent_actions = valid_actions[-8:]
        inferred_inventory = self._path_opd_inventory_from_actions(valid_actions)
        visited_locations = self._path_opd_visited_locations(valid_actions)
        repeated_actions = self._path_opd_repeated_actions(valid_actions)
        reminders: List[str] = []
        if repeated_actions:
            reminders.append("Avoid unproductive retries listed in repeated_action_history.")
        if last_invalid_action:
            reminders.append("The invalid_action failed in the immediately previous context; retry only if the state has changed.")

        lines = ["[History Summary]"]
        lines.append("This is a concise execution-history summary for the current ALFWorld episode.")
        lines.append(f"current_location: {current_location}")
        if inferred_inventory:
            lines.append(f"inferred_inventory_from_action_history: {inferred_inventory}")
        lines.append(f"visited_locations: {visited_locations if visited_locations else []}")
        lines.append(f"recent_action_history: {recent_actions if recent_actions else []}")
        if repeated_actions:
            lines.append(f"repeated_action_history: {repeated_actions}")
        if last_invalid_action:
            lines.append(f"invalid_action: {last_invalid_action}")
        if reminders:
            lines.append(f"state_reminder: {' '.join(reminders)}")
        lines.append("[/History Summary]")
        return "\n".join(lines)

    @staticmethod
    def _vmpr_is_clean_action(action: Any) -> bool:
        text = str(action or "").strip()
        if not text:
            return False
        lowered = text.lower()
        if "\n" in text or "\r" in text or "<" in text or ">" in text or "```" in text:
            return False
        if "action:" in lowered or "assistant" in lowered or "user" in lowered or "person" in lowered:
            return False
        # ALFWorld actions stay under the short limit. WebShop search[...] queries are
        # often longer than 120 chars; only relax the cap for that prefix so ALFWorld
        # cleanup behavior is unchanged.
        max_len = 512 if lowered.startswith("search[") else 120
        if len(text) > max_len:
            return False
        return True

    @staticmethod
    def _vmpr_admissible_list_looks_webshop(prompt_admissible: List[str]) -> bool:
        """Detect WebShop admissible lists without touching ALFWorld exact-match cleanup."""
        for candidate in prompt_admissible:
            norm = normalize_action_text(candidate)
            if norm == "search[<your query>]" or norm.startswith("click[buy now]"):
                return True
        return False

    @classmethod
    def _vmpr_action_is_prompt_admissible(cls, action: str, prompt_admissible: List[str]) -> bool:
        """Return whether ``action`` is allowed by the current prompt's admissible list.

        ALFWorld keeps strict normalized-string membership. WebShop admissible lists
        expose ``search[<your query>]`` as a placeholder, so exact membership would
        drop every real ``search[...]`` from student success paths.
        """
        if not prompt_admissible:
            return True
        if cls._vmpr_admissible_list_looks_webshop(prompt_admissible):
            return webshop_action_is_admissible(action, prompt_admissible)
        return normalize_action_text(action) in {normalize_action_text(candidate) for candidate in prompt_admissible}

    @staticmethod
    def _vmpr_valid_action_flag(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, float, np.integer, np.floating)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "valid"}:
            return True
        if text in {"0", "false", "no", "invalid", ""}:
            return False
        return bool(value)

    @classmethod
    def _vmpr_clean_trajectory_actions(cls, traj: List[Dict[str, Any]], *, filter_no_progress: bool = True) -> Tuple[List[str], int, int, int, int]:
        by_turn: Dict[int, str] = {}
        observations_by_turn: Dict[int, str] = {}
        dropped = 0
        duplicates = 0
        no_progress = 0
        non_admissible = 0
        for order, item in enumerate(traj):
            if not item.get("active_masks"):
                continue
            try:
                turn = int(item.get("turn_step", order) or 0)
            except Exception:
                turn = order
            if turn not in observations_by_turn:
                current_observation = str(item.get("vmpr_current_observation") or "")
                if not current_observation:
                    prompt_text = str(item.get("input") or "")
                    current_observation = extract_current_observation(prompt_text) or prompt_text
                observations_by_turn[turn] = current_observation

        for order, item in enumerate(traj):
            if not item.get("active_masks"):
                continue
            try:
                turn = int(item.get("turn_step", order) or 0)
            except Exception:
                turn = order
            action = str(item.get("executed_action") or "").strip()
            valid = cls._vmpr_valid_action_flag(item.get("is_action_valid", True))
            if not valid or not cls._vmpr_is_clean_action(action):
                if action:
                    dropped += 1
                continue
            if turn in by_turn:
                duplicates += 1
                continue
            next_observation = str(item.get("vmpr_next_observation") or observations_by_turn.get(turn + 1, ""))
            action_made_no_progress = bool(filter_no_progress and next_observation and "nothing happens" in normalize_observation_text(next_observation))
            prompt_admissible_value = item.get("vmpr_admissible_actions")
            if isinstance(prompt_admissible_value, np.ndarray):
                prompt_admissible_value = prompt_admissible_value.tolist()
            if not isinstance(prompt_admissible_value, (list, tuple)):
                prompt_admissible_value = []
            prompt_admissible = [str(candidate) for candidate in prompt_admissible_value]
            if not prompt_admissible:
                prompt_admissible = extract_admissible_actions(str(item.get("input") or ""))
            action_non_admissible = bool(prompt_admissible) and not cls._vmpr_action_is_prompt_admissible(action, prompt_admissible)
            if action_made_no_progress:
                no_progress += 1
            if action_non_admissible:
                non_admissible += 1
            if action_made_no_progress or action_non_admissible:
                continue
            by_turn[turn] = action
        return [action for _, action in sorted(by_turn.items(), key=lambda pair: pair[0])], dropped, duplicates, no_progress, non_admissible

    def _vmpr_prefix_len_bounds(self, global_step: int) -> Tuple[Optional[int], Optional[int]]:
        configured_min = self.vmpr_cfg.get("min_prefix_len", None)
        configured_max = self.vmpr_cfg.get("max_prefix_len", None)
        min_prefix_len = None if configured_min is None else int(configured_min)
        max_prefix_len = None if configured_max is None else int(configured_max)
        schedule = self.vmpr_cfg.get("prefix_len_schedule", None)
        if schedule != "tcod_b2f_linear":
            return min_prefix_len, max_prefix_len

        checkpoint_steps = max(1, int(self.vmpr_cfg.get("tcod_checkpoint_steps", 5)))
        max_env_steps = max(0, int(self.vmpr_cfg.get("tcod_max_env_steps", self.config.env.max_steps)))
        # TCOD B2F uses checkpoint_steps as the interval for reducing the
        # expert roll-in by one action, not as the amount to subtract.
        target_len = max(0, (max_env_steps - 1) - (global_step // checkpoint_steps))
        self._vmpr_pending_metrics["vmpr/target_prefix_len"] = float(target_len)
        if target_len <= 0:
            return 1, 0
        if max_prefix_len is not None:
            target_len = min(target_len, max_prefix_len)
        return min_prefix_len, target_len

    def _vmpr_reset_metrics(self, *, global_step: int = 0) -> Dict[str, Any]:
        metrics = {
            "vmpr/enabled": float(self.vmpr_enabled),
            "vmpr/prefix_start_sampled": 0.0,
            "vmpr/full_start_sampled": 0.0,
            "vmpr/replay_success": 0.0,
            "vmpr/replay_fail": 0.0,
            "vmpr/added_student_prefixes": 0.0,
            "vmpr/skipped_group_all_success_trajectories": 0.0,
            "vmpr/skipped_quality_success_trajectories": 0.0,
            "vmpr/success_path_quality_checked_trajectories": 0.0,
        }
        if self.prefix_buffer is not None:
            metrics["vmpr/buffer_size"] = float(len(self.prefix_buffer))
            for source, count in self.prefix_buffer.source_counts().items():
                metrics[f"vmpr/buffer_{source}"] = float(count)
            for prefix_len, count in self.prefix_buffer.prefix_len_counts().items():
                metrics[f"vmpr/buffer_prefix_len_{prefix_len}"] = float(count)
        return metrics

    def _normalize_env_kwargs(self, env_kwargs: Any, batch_size: int) -> List[Dict[str, Any]]:
        if env_kwargs is None:
            return [{} for _ in range(batch_size)]
        if isinstance(env_kwargs, dict):
            return [dict(env_kwargs) for _ in range(batch_size)]
        kwargs_list = list(env_kwargs)
        if len(kwargs_list) != batch_size:
            raise ValueError(f"Expected {batch_size} env kwargs, got {len(kwargs_list)}")
        return [dict(item or {}) for item in kwargs_list]

    def _vmpr_prepare_gen_batch(self, gen_batch: DataProto, *, global_step: int, is_train: bool) -> DataProto:
        self._vmpr_pending_metrics = self._vmpr_reset_metrics(global_step=global_step)
        if not self.vmpr_enabled or not is_train or self.prefix_buffer is None:
            return gen_batch

        batch_size = len(gen_batch.batch)
        group_n = max(1, int(self.config.env.rollout.n))
        env_kwargs = self._normalize_env_kwargs(gen_batch.non_tensor_batch.get("env_kwargs"), batch_size)
        vmpr_group_uid = [None for _ in range(batch_size)]
        prefix_start_ratio = float(self.vmpr_cfg.get("prefix_start_ratio", 0.0))
        min_prefix_len, max_prefix_len = self._vmpr_prefix_len_bounds(global_step)
        sampled_prefix_records: List[Tuple[Dict[str, Any], int]] = []

        for group_start in range(0, batch_size, group_n):
            group_end = min(group_start + group_n, batch_size)
            entry = None
            if prefix_start_ratio > 0 and self.vmpr_rng.random() < prefix_start_ratio and (max_prefix_len is None or max_prefix_len >= 1):
                entry = self.prefix_buffer.sample(
                    global_step=global_step,
                    min_prefix_len=min_prefix_len,
                    max_prefix_len=max_prefix_len,
                    prefer_longest_below_max=self.vmpr_cfg.get("prefix_len_schedule", None) == "tcod_b2f_linear",
                )
                if entry is not None and self.prefix_buffer.last_sample_stats is not None:
                    sampled_prefix_records.append((dict(self.prefix_buffer.last_sample_stats), group_end - group_start))

            if entry is None:
                uid = str(uuid.uuid4())
                self._vmpr_pending_metrics["vmpr/full_start_sampled"] += group_end - group_start
                for idx in range(group_start, group_end):
                    env_kwargs[idx].update(
                        {
                            "vmpr_prefix_id": None,
                            "vmpr_source": FULL_START_SOURCE,
                            "prefix_actions": [],
                            "vmpr_prefix_len": 0,
                        }
                    )
                    vmpr_group_uid[idx] = uid
                continue

            uid = f"vmpr:{entry.prefix_id}:{uuid.uuid4()}"
            reset_kwargs = entry.to_reset_kwargs()
            self._vmpr_pending_metrics["vmpr/prefix_start_sampled"] += group_end - group_start
            self._vmpr_pending_metrics[f"vmpr/prefix_sampled_{entry.source}"] = self._vmpr_pending_metrics.get(f"vmpr/prefix_sampled_{entry.source}", 0.0) + group_end - group_start
            self._vmpr_pending_metrics[f"vmpr/prefix_len_{entry.prefix_len}_sampled"] = self._vmpr_pending_metrics.get(f"vmpr/prefix_len_{entry.prefix_len}_sampled", 0.0) + group_end - group_start
            for idx in range(group_start, group_end):
                merged = dict(env_kwargs[idx])
                merged.update(reset_kwargs)
                merged["vmpr_prefix_len"] = entry.prefix_len
                env_kwargs[idx] = merged
                vmpr_group_uid[idx] = uid

        prefix_start_count = self._vmpr_pending_metrics["vmpr/prefix_start_sampled"]
        full_start_count = self._vmpr_pending_metrics["vmpr/full_start_sampled"]
        self._vmpr_pending_metrics["vmpr/prefix_start_ratio_realized"] = prefix_start_count / max(1.0, prefix_start_count + full_start_count)
        self._vmpr_add_sampled_prefix_metrics(sampled_prefix_records)
        gen_batch.non_tensor_batch["env_kwargs"] = np.array(env_kwargs, dtype=object)
        gen_batch.non_tensor_batch["vmpr_group_uid"] = np.array(vmpr_group_uid, dtype=object)
        return gen_batch

    def _vmpr_add_sampled_prefix_metrics(self, records: List[Tuple[Dict[str, Any], int]]) -> None:
        if not records:
            return

        def add_metrics(prefix: str, selected: List[Tuple[Dict[str, Any], int]]) -> None:
            total_weight = sum(weight for _, weight in selected)
            if total_weight <= 0:
                return
            for field in ("age", "idle_steps", "use_count_before"):
                value = sum(float(stats.get(field, 0.0)) * weight for stats, weight in selected) / total_weight
                self._vmpr_pending_metrics[f"{prefix}_{field}_mean"] = value
            task_keys = {str(stats.get("task_key")) for stats, _ in selected if stats.get("task_key")}
            self._vmpr_pending_metrics[f"{prefix}_unique_tasks"] = float(len(task_keys))
            self._vmpr_pending_metrics[f"{prefix}_unique_task_ratio"] = len(task_keys) / max(1, len(selected))

        add_metrics("vmpr/sampled_prefix", records)
        for source in (TEACHER_SOURCE, STUDENT_SOURCE, ASSISTED_SOURCE):
            source_records = [(stats, weight) for stats, weight in records if stats.get("source") == source]
            if source_records:
                add_metrics(f"vmpr/sampled_prefix_{source}", source_records)

    def _vmpr_info_array(self, infos: List[Dict[str, Any]], key: str, default: Any = None, dtype=object) -> np.ndarray:
        return np.array([info.get(key, default) for info in infos], dtype=dtype)

    def _vmpr_action_list(self, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                value = value.item()
            else:
                value = value.tolist()
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple)):
            return [str(action) for action in value if action]
        return [str(value)]

    def _vmpr_success_values(self, success: Dict[str, np.ndarray], rewards: np.ndarray, batch_size: int) -> np.ndarray:
        values = success.get("success_rate")
        if values is None or len(values) != batch_size:
            return np.asarray(rewards > 0, dtype=bool)
        return np.asarray(values, dtype=float) > 0

    def _vmpr_task_key_and_reset_kwargs(self, infos: List[Dict[str, Any]]) -> tuple[str, Dict[str, Any]]:
        task_key = ""
        reset_kwargs: Dict[str, Any] = {}
        for info in reversed(infos):
            gamefile = info.get("extra.gamefile")
            session_id = info.get("session_id")
            if gamefile:
                task_key = str(gamefile)
                reset_kwargs = {"gamefile": gamefile}
                break
            if session_id is not None:
                # WebShop keys candidate lookup by the stable content-hash task id
                # (``webshop_task_id``), which is what ``batch["task_key"]`` carries at
                # retrieval time. The numeric ``session_id``/goal index is only needed to
                # reset the right goal, so keep it in ``reset_kwargs`` but use the same
                # logical task id as the buffer key. Otherwise the buffer is keyed by the
                # numeric session while retrieval queries the hash and never matches.
                logical_task_key = info.get("task_key") or info.get("webshop_task_id")
                task_key = str(logical_task_key) if logical_task_key else str(session_id)
                reset_kwargs = {"session_id": session_id}
                break
        return task_key, reset_kwargs

    def _vmpr_success_index(self) -> Optional[PathTrajectoryIndex]:
        if self.vmpr_success_max_length_ratio_to_canonical <= 0:
            return None
        if self._vmpr_success_trajectory_index is None:
            self._vmpr_success_trajectory_index = PathTrajectoryIndex.from_jsonl(self.vmpr_success_trajectory_index_path)
        return self._vmpr_success_trajectory_index

    def _vmpr_success_path_quality_max_len(self, task_key: str, reset_kwargs: Dict[str, Any]) -> int:
        max_len = 0
        index = self._vmpr_success_index()
        if index is not None:
            canonical_actions = index.get(reset_kwargs.get("gamefile"), task_key, reset_kwargs.get("session_id"))
            if canonical_actions:
                max_len = max(1, int(math.ceil(len(canonical_actions) * self.vmpr_success_max_length_ratio_to_canonical)))
        if self.vmpr_success_max_action_len > 0:
            max_len = min(max_len, self.vmpr_success_max_action_len) if max_len > 0 else self.vmpr_success_max_action_len
        return max_len

    def _vmpr_update_prefix_buffer(
        self,
        *,
        total_batch_list: List[List[Dict]],
        total_infos: List[List[Dict]],
        episode_rewards: np.ndarray,
        success: Dict[str, np.ndarray],
        global_step: int,
        is_train: bool,
    ) -> None:
        if not self.vmpr_enabled or self.prefix_buffer is None or not is_train:
            return
        if len(total_infos) != len(total_batch_list):
            return

        success_values = self._vmpr_success_values(success, episode_rewards, len(total_batch_list))
        rollout_stats: Dict[str, Dict[str, float]] = {}
        for traj_idx, traj in enumerate(total_batch_list):
            if not traj:
                continue
            source = str(traj[0].get("vmpr_source") or FULL_START_SOURCE)
            stats = rollout_stats.setdefault(source, {"count": 0.0, "success": 0.0, "suffix_steps": 0.0})
            stats["count"] += 1.0
            stats["success"] += float(bool(success_values[traj_idx]))
            stats["suffix_steps"] += sum(1.0 for item in traj if item.get("active_masks"))
        for source, stats in rollout_stats.items():
            count = max(1.0, stats["count"])
            self._vmpr_pending_metrics[f"vmpr/rollout_trajectories_{source}"] = stats["count"]
            self._vmpr_pending_metrics[f"vmpr/rollout_success_rate_{source}"] = stats["success"] / count
            self._vmpr_pending_metrics[f"vmpr/suffix_steps_mean_{source}"] = stats["suffix_steps"] / count

        skip_group_all_success = bool(self.vmpr_cfg.get("skip_add_when_group_success_all", False))
        skip_group_success_ratio_above = float(self.vmpr_cfg.get("skip_add_when_group_success_ratio_above", 0.0) or 0.0)
        if skip_group_success_ratio_above < 0 or skip_group_success_ratio_above >= 1:
            raise ValueError("skip_add_when_group_success_ratio_above must be in [0, 1).")
        group_best_success_only = bool(self.vmpr_cfg.get("group_best_success_only", False))
        group_success_values: Dict[str, List[bool]] = {}
        group_success_indices: Dict[str, List[int]] = {}
        if skip_group_all_success or skip_group_success_ratio_above > 0 or group_best_success_only:
            for traj_idx, traj in enumerate(total_batch_list):
                if not traj:
                    continue
                uid = str(traj[0].get("uid") or traj[0].get("traj_uid") or traj_idx)
                group_success_values.setdefault(uid, []).append(bool(success_values[traj_idx]))
                if bool(success_values[traj_idx]) and not bool(traj[0].get("vmpr_is_prefix_start")):
                    group_success_indices.setdefault(uid, []).append(traj_idx)

        group_best_success_indices = set()
        if group_best_success_only:
            for indices in group_success_indices.values():
                if not indices:
                    continue
                best_idx = min(
                    indices,
                    key=lambda idx: (
                        sum(1 for item in total_batch_list[idx] if item.get("active_masks") and item.get("executed_action")),
                        idx,
                    ),
                )
                group_best_success_indices.add(best_idx)

        for traj_idx, traj in enumerate(total_batch_list):
            if not traj:
                continue
            first = traj[0]
            prefix_id = first.get("vmpr_prefix_id")
            is_prefix_start = bool(first.get("vmpr_is_prefix_start"))
            won = bool(success_values[traj_idx])
            if is_prefix_start:
                self.prefix_buffer.update_success(prefix_id, success=won)
            if not won:
                continue
            if skip_group_all_success:
                uid = str(first.get("uid") or first.get("traj_uid") or traj_idx)
                values = group_success_values.get(uid, [])
                if values and all(values):
                    self._vmpr_pending_metrics["vmpr/skipped_group_all_success_trajectories"] += 1.0
                    continue
            if skip_group_success_ratio_above > 0 and not is_prefix_start:
                uid = str(first.get("uid") or first.get("traj_uid") or traj_idx)
                values = group_success_values.get(uid, [])
                success_ratio = sum(values) / len(values) if values else 0.0
                if success_ratio > skip_group_success_ratio_above:
                    self._vmpr_pending_metrics["vmpr/skipped_group_high_success_trajectories"] = self._vmpr_pending_metrics.get("vmpr/skipped_group_high_success_trajectories", 0.0) + 1.0
                    continue
            if group_best_success_only and not is_prefix_start and traj_idx not in group_best_success_indices:
                self._vmpr_pending_metrics["vmpr/skipped_non_best_success_trajectories"] = self._vmpr_pending_metrics.get("vmpr/skipped_non_best_success_trajectories", 0.0) + 1.0
                continue
            if is_prefix_start and not bool(self.vmpr_cfg.get("update_assisted_prefixes", True)):
                continue
            if not is_prefix_start and not bool(self.vmpr_cfg.get("update_student_prefixes", True)):
                continue

            filter_no_progress_success_actions = bool(self.vmpr_cfg.get("filter_no_progress_success_actions", True))
            suffix_actions, dropped_actions, duplicate_turns, no_progress_actions, non_admissible_actions = self._vmpr_clean_trajectory_actions(
                traj, filter_no_progress=filter_no_progress_success_actions
            )
            if dropped_actions:
                self._vmpr_pending_metrics["vmpr/dropped_unclean_success_actions"] = self._vmpr_pending_metrics.get("vmpr/dropped_unclean_success_actions", 0.0) + float(dropped_actions)
            if duplicate_turns:
                self._vmpr_pending_metrics["vmpr/dropped_duplicate_success_turns"] = self._vmpr_pending_metrics.get("vmpr/dropped_duplicate_success_turns", 0.0) + float(duplicate_turns)
            if no_progress_actions:
                self._vmpr_pending_metrics["vmpr/dropped_no_progress_success_actions"] = self._vmpr_pending_metrics.get("vmpr/dropped_no_progress_success_actions", 0.0) + float(no_progress_actions)
            if non_admissible_actions:
                self._vmpr_pending_metrics["vmpr/dropped_non_admissible_success_actions"] = self._vmpr_pending_metrics.get("vmpr/dropped_non_admissible_success_actions", 0.0) + float(non_admissible_actions)
            if not suffix_actions:
                continue
            task_key, reset_kwargs = self._vmpr_task_key_and_reset_kwargs(total_infos[traj_idx])
            source_metadata = {
                "source_traj_uid": str(first.get("traj_uid") or ""),
            }
            prefix_actions = self._vmpr_action_list(first.get("vmpr_replayed_actions")) if is_prefix_start else []
            write_actions = prefix_actions + suffix_actions if is_prefix_start else suffix_actions
            quality_max_len = self._vmpr_success_path_quality_max_len(task_key, reset_kwargs)
            if quality_max_len > 0:
                self._vmpr_pending_metrics["vmpr/success_path_quality_max_len_sum"] = self._vmpr_pending_metrics.get("vmpr/success_path_quality_max_len_sum", 0.0) + float(quality_max_len)
                self._vmpr_pending_metrics["vmpr/success_path_quality_checked_trajectories"] = self._vmpr_pending_metrics.get("vmpr/success_path_quality_checked_trajectories", 0.0) + 1.0
                if len(write_actions) > quality_max_len:
                    self._vmpr_pending_metrics["vmpr/skipped_quality_success_trajectories"] = self._vmpr_pending_metrics.get("vmpr/skipped_quality_success_trajectories", 0.0) + 1.0
                    self._vmpr_pending_metrics["vmpr/skipped_quality_success_path_len_sum"] = self._vmpr_pending_metrics.get("vmpr/skipped_quality_success_path_len_sum", 0.0) + float(len(write_actions))
                    continue
            if is_prefix_start:
                assisted_metadata = {
                    **source_metadata,
                    "parent_prefix_id": prefix_id,
                    "parent_source": first.get("vmpr_source"),
                }
                added = self.prefix_buffer.add_trajectory_prefixes(
                    source=ASSISTED_SOURCE,
                    env_name=self.config.env.env_name,
                    task_key=task_key,
                    reset_kwargs=reset_kwargs,
                    actions=write_actions,
                    global_step=global_step,
                    metadata=assisted_metadata,
                )
                self._vmpr_pending_metrics["vmpr/added_assisted_prefixes"] = self._vmpr_pending_metrics.get("vmpr/added_assisted_prefixes", 0.0) + added
            else:
                added = self.prefix_buffer.add_trajectory_prefixes(
                    source=STUDENT_SOURCE,
                    env_name=self.config.env.env_name,
                    task_key=task_key,
                    reset_kwargs=reset_kwargs,
                    actions=write_actions,
                    global_step=global_step,
                    metadata=source_metadata,
                )
                self._vmpr_pending_metrics["vmpr/added_student_prefixes"] += added
            duplicate_count = float(self.prefix_buffer.last_add_duplicate_count)
            if duplicate_count:
                self._vmpr_pending_metrics["vmpr/skipped_duplicate_prefixes"] = self._vmpr_pending_metrics.get("vmpr/skipped_duplicate_prefixes", 0.0) + duplicate_count

        save_every = int(self.vmpr_cfg.get("save_every_steps", 1) or 0)
        if save_every > 0 and global_step % save_every == 0:
            self.prefix_buffer.save()

        self._vmpr_pending_metrics["vmpr/buffer_size"] = float(len(self.prefix_buffer))
        for source, count in self.prefix_buffer.source_counts().items():
            self._vmpr_pending_metrics[f"vmpr/buffer_{source}"] = float(count)
        for prefix_len, count in self.prefix_buffer.prefix_len_counts().items():
            self._vmpr_pending_metrics[f"vmpr/buffer_prefix_len_{prefix_len}"] = float(count)
        checked = float(self._vmpr_pending_metrics.get("vmpr/success_path_quality_checked_trajectories", 0.0))
        if checked > 0:
            skipped = float(self._vmpr_pending_metrics.get("vmpr/skipped_quality_success_trajectories", 0.0))
            self._vmpr_pending_metrics["vmpr/skipped_quality_success_ratio"] = skipped / checked
            self._vmpr_pending_metrics["vmpr/success_path_quality_max_len_mean"] = float(self._vmpr_pending_metrics.get("vmpr/success_path_quality_max_len_sum", 0.0)) / checked
            if skipped > 0:
                self._vmpr_pending_metrics["vmpr/skipped_quality_success_path_len_mean"] = float(self._vmpr_pending_metrics.get("vmpr/skipped_quality_success_path_len_sum", 0.0)) / skipped

    def preprocess_single_sample(
        self,
        item: int,
        gen_batch: DataProto,
        obs: Dict,
        chat_histories: Optional[List[List[Dict[str, str]]]] = None,
        state_histories: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Process a single observation sample, organizing environment observations (text and/or images) 
        into a format processable by the model.
        
        Parameters:
            item (int): Sample index in the batch
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation, may contain 'text', 'image', 'anchor' keys
        
        Returns:
            dict: Contains processed input data such as input_ids, attention_mask, etc.
        """

        raw_prompt = gen_batch.non_tensor_batch['raw_prompt'][item]
        data_source = gen_batch.non_tensor_batch['data_source'][item]
        apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        
        # Get observation components
        obs_texts = obs.get('text', None)
        obs_images = obs.get('image', None)
        obs_anchors = obs.get('anchor', None)
        obs_text = obs_texts[item] if obs_texts is not None else None
        compact_chat_texts = obs.get('compact_chat_text', None)
        obs_image = obs_images[item] if obs_images is not None else None
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        is_multi_modal = obs_image is not None

        _obs_anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor

        # Build chat structure
        # obs_content = raw_prompt[0]['content']
        # if '<image>' in obs_content: 
        #     obs_content = obs_content.replace('<image>', '')

        # Build chat structure
        obs_content = ''
        if obs_text is not None:
            obs_content += obs_text
        else:
            print(f"Warning: No text observation found!")
        chat_history_prompt_mode = self.config.env.alfworld.get("chat_history_prompt_mode", "tcod_repeated")
        if chat_histories is not None and chat_histories[item] and chat_history_prompt_mode == "compact_react":
            if compact_chat_texts is None:
                raise ValueError("compact_react chat history requires compact_chat_text observations")
            obs_content = compact_chat_texts[item]
        if self.path_opd_student_context == "state_history":
            if state_histories is None:
                raise ValueError("algorithm.path_opd.student_context=state_history requires rollout state histories")
            state_block = self._path_opd_state_block_from_rollout_history(item, obs_content, state_histories[item])
            obs_content = self._path_opd_insert_context(obs_content, state_block)

        
        current_user_message = {
            "content": obs_content,
            "role": "user",
        }
        chat = list(chat_histories[item]) if chat_histories is not None else []
        chat.append(current_user_message)
        
        # Apply chat template
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
            **apply_chat_template_kwargs
        )
        chat_history_dropped_turns = 0
        if chat_histories is not None:
            prompt_token_count = len(self.tokenizer.encode(prompt_with_chat_template, add_special_tokens=False))
            while prompt_token_count > self.config.data.max_prompt_length and len(chat) >= 3:
                # Keep the current user observation and remove complete oldest
                # suffix turns so the chat roles remain well formed.
                if chat[0].get("role") != "user" or chat[1].get("role") != "assistant":
                    break
                chat = chat[2:]
                chat_history_dropped_turns += 1
                prompt_with_chat_template = self.tokenizer.apply_chat_template(
                    chat,
                    add_generation_prompt=True,
                    tokenize=False,
                    **apply_chat_template_kwargs
                )
                prompt_token_count = len(self.tokenizer.encode(prompt_with_chat_template, add_special_tokens=False))
        
        # Initialize return dict
        row_dict = {}
        
        # Process multimodal data
        if is_multi_modal:
            # Replace image placeholder with vision tokens
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(obs_image)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                                self.processor.image_token)

        else:
            raw_prompt = prompt_with_chat_template
        
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                            tokenizer=self.tokenizer,
                                                                            max_length=self.config.data.max_prompt_length,
                                                                            pad_token_id=self.tokenizer.pad_token_id,
                                                                            left_pad=True,
                                                                            truncation=self.config.data.truncation,)
        
        

        if is_multi_modal:

            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.config.data.max_prompt_length:
            if self.config.data.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.config.data.max_prompt_length :]
            elif self.config.data.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.config.data.max_prompt_length]
            elif self.config.data.truncation == "middle":
                left_half = self.config.data.max_prompt_length // 2
                right_half = self.config.data.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.config.data.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.config.data.max_prompt_length}.")

        # Build final output dict
        row_dict.update({
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt_ids': raw_prompt_ids,
            'vmpr_current_observation': extract_current_observation(obs_content) or obs_content,
            'vmpr_admissible_actions': extract_admissible_actions(obs_content),
            'anchor_obs': _obs_anchor,
            'index': item,
            'data_source': data_source,
            'chat_history_dropped_turns': chat_history_dropped_turns,
        })
        for key in ("validation_item_id", "validation_sample_index"):
            if key in gen_batch.non_tensor_batch:
                row_dict[key] = gen_batch.non_tensor_batch[key][item]

        if self.config.data.get('return_raw_chat', False):
            row_dict['raw_prompt'] = chat
        
        return row_dict

    def preprocess_batch(
        self,
        gen_batch: DataProto, 
        obs: Dict, 
        chat_histories: Optional[List[List[Dict[str, str]]]] = None,
        state_histories: Optional[List[Dict[str, Any]]] = None,
    ) -> DataProto:
        """
        Process a batch of observation samples, converting environment observations into model-processable format.
        
        Parameters:
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation dictionary
                - 'text' (None or List[str]): Text observation data
                - 'image' (np.ndarray or torch.Tensor): Image observation data
                - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
        
        Returns:
            DataProto: Contains processed batch data with preserved metadata
        """
        batch_size = len(gen_batch.batch['input_ids'])
        processed_samples = []
        
        # Process each sample in parallel
        for item in range(batch_size):
            # Extract per-sample observations
            processed = self.preprocess_single_sample(
                item=item,
                gen_batch=gen_batch,
                obs=obs,
                chat_histories=chat_histories,
                state_histories=state_histories,
            )
            processed_samples.append(processed)
        
        # Aggregate batch data
        batch = collate_fn(processed_samples)
        
        # Create DataProto with preserved metadata
        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        return new_batch


    def gather_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            tool_callings: np.ndarray,
            ) -> DataProto:
        """
        Collect and organize trajectory data, handling batch size adjustments to meet parallel training requirements.
        
        Parameters:
            total_batch_list (List[List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
            tool_callings (np.ndarray): Number of tool callings for each environment
        Returns:
            DataProto: Collected and organized trajectory data
        """
        batch_size = len(total_batch_list)

        success_rate = {}
        for key, value in success.items():
            success_rate[key] = np.mean(value)
        
        effective_batch = []
        for bs in range(batch_size):
            # sum the rewards for each data in total_batch_list[bs]
            for step_idx, data in enumerate(total_batch_list[bs]):
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                if data['active_masks']:
                    # episode_rewards
                    data['episode_rewards'] = episode_rewards[bs]
                    # episode_lengths
                    data['episode_lengths'] = episode_lengths[bs]
                    # tool_callings
                    data['tool_callings'] = tool_callings[bs]
                    # turn_step: which step within the trajectory
                    data['turn_step'] = step_idx
                    # success_rate
                    for key, value in success_rate.items():
                        data[key] = value

                    effective_batch.append(data)
            
        # Convert trajectory data to DataProto format
        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )
        return gen_batch_output

    def vanilla_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Collects trajectories through parallel agent-environment agent_loop.
        Parameters:
            gen_batch (DataProto): Initial batch with prompts to start the agent_loop
            actor_rollout_wg (WorkerGroup): Worker group containing the actor model for policy decisions
            envs (EnvironmentManagerBase): Environment manager containing parallel environment instances
        
        Returns:
            total_batch_list (List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
        """

        batch_size = len(gen_batch.batch)

        # Initial observations from the environment
        raw_env_kwargs = gen_batch.non_tensor_batch.get('env_kwargs', None)
        reset_kwargs = self._normalize_env_kwargs(raw_env_kwargs, batch_size)
        obs, infos = envs.reset(kwargs=None if raw_env_kwargs is None else reset_kwargs)

        reset_prefix_actions = [list(item.get("prefix_actions") or []) for item in reset_kwargs]
        reset_prefix_id = [info.get("vmpr_prefix_id") for info in infos]
        reset_source = [info.get("vmpr_source", FULL_START_SOURCE) for info in infos]
        reset_prefix_len = [int(info.get("vmpr_prefix_len", len(reset_prefix_actions[i]) if reset_prefix_actions else 0) or 0) for i, info in enumerate(infos)]
        reset_replay_success = [bool(info.get("vmpr_replay_success", True)) for info in infos]
        reset_metadata = [info.get("vmpr_metadata") or {} for info in infos]
        reset_is_prefix_start = [bool(reset_prefix_id[i]) and reset_source[i] != FULL_START_SOURCE and reset_replay_success[i] for i in range(len(infos))]
        if self.vmpr_enabled and self.prefix_buffer is not None:
            for info in infos:
                failed_prefix_id = info.get("vmpr_failed_prefix_id")
                if failed_prefix_id:
                    self.prefix_buffer.mark_replay_result(failed_prefix_id, success=False)
                    self._vmpr_pending_metrics["vmpr/replay_fail"] = self._vmpr_pending_metrics.get("vmpr/replay_fail", 0.0) + 1.0
                prefix_id = info.get("vmpr_prefix_id")
                if prefix_id and info.get("vmpr_replay_success", True):
                    self.prefix_buffer.mark_replay_result(prefix_id, success=True)
                    self._vmpr_pending_metrics["vmpr/replay_success"] = self._vmpr_pending_metrics.get("vmpr/replay_success", 0.0) + 1.0

        lenght_obs = len(obs['text']) if obs['text'] is not None else len(obs['image'])
        assert len(gen_batch.batch) == lenght_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {lenght_obs}"
        
        if "vmpr_group_uid" in gen_batch.non_tensor_batch:
            uid_batch = np.array(gen_batch.non_tensor_batch["vmpr_group_uid"], dtype=object)
        elif self.config.env.rollout.n > 0: # env grouping
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else: # no env grouping, set all to the same uid
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)
        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)
        accumulate_chat_history = bool(self.config.env.alfworld.get("accumulate_chat_history", False))
        chat_histories: Optional[List[List[Dict[str, str]]]] = [[] for _ in range(batch_size)] if accumulate_chat_history else None
        state_histories: Optional[List[Dict[str, Any]]] = None
        if self.path_opd_student_context == "state_history":
            state_histories = [
                {
                    "valid_actions": [],
                    "no_progress_actions": [],
                    "malformed_actions": [],
                    "last_invalid_action": "",
                    "last_location": infer_location_from_observation(str(obs["text"][i] if obs.get("text", None) is not None else "")),
                }
                for i in range(batch_size)
            ]
        # Trajectory collection loop
        for _step in range(self.config.env.max_steps):
            active_masks = np.logical_not(is_done)

            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs, chat_histories=chat_histories, state_histories=state_histories)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            batch_input.meta_info = gen_batch.meta_info

            # pad to be divisible by dp_size
            batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
            batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
            # # unpad
            batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)

            batch.non_tensor_batch['uid'] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid
            batch.non_tensor_batch['vmpr_prefix_id'] = np.array(reset_prefix_id, dtype=object)
            batch.non_tensor_batch['vmpr_source'] = np.array(reset_source, dtype=object)
            batch.non_tensor_batch['vmpr_prefix_len'] = np.array(reset_prefix_len, dtype=np.int32)
            batch.non_tensor_batch['vmpr_replay_success'] = np.array(reset_replay_success, dtype=bool)
            batch.non_tensor_batch['vmpr_is_prefix_start'] = np.array(reset_is_prefix_start, dtype=bool)
            batch.non_tensor_batch['vmpr_replayed_actions'] = np.array(reset_prefix_actions, dtype=object)
            batch.non_tensor_batch['vmpr_metadata'] = np.array(reset_metadata, dtype=object)

            batch = batch.union(batch_output)
            
            text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)
            raw_text_actions = list(text_actions)

            if chat_histories is not None:
                for i in range(batch_size):
                    if active_masks[i]:
                        user_content = obs["text"][i]
                        if (
                            chat_histories[i]
                            and self.config.env.alfworld.get("chat_history_prompt_mode", "tcod_repeated") == "compact_react"
                        ):
                            user_content = obs["compact_chat_text"][i]
                        chat_histories[i].append({"role": "user", "content": user_content})
                        chat_histories[i].append({"role": "assistant", "content": raw_text_actions[i]})
            
            env_step_actions = list(raw_text_actions)
            next_obs, rewards, dones, infos = envs.step(env_step_actions)

            
            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                # dones is numpy, delete a dimension
                dones = dones.squeeze(1)

            if 'is_action_valid' in infos[0]:
                env_action_valid = np.array([info['is_action_valid'] for info in infos], dtype=bool)
            else:
                env_action_valid = np.ones(batch_size, dtype=bool)
            extra_output_after_action = np.array(
                [self._has_extra_output_after_first_action(action) for action in raw_text_actions],
                dtype=bool,
            )
            if self.penalize_extra_output_after_action:
                action_valid_for_penalty = np.logical_and(env_action_valid, np.logical_not(extra_output_after_action))
            else:
                action_valid_for_penalty = env_action_valid
            batch.non_tensor_batch['env_is_action_valid'] = env_action_valid
            batch.non_tensor_batch['has_extra_output_after_action'] = extra_output_after_action
            batch.non_tensor_batch['is_action_valid'] = action_valid_for_penalty
            batch.non_tensor_batch['executed_action'] = np.array([info.get('executed_action', '') for info in infos], dtype=object)
            batch.non_tensor_batch['gamefile'] = np.array([info.get('extra.gamefile') for info in infos], dtype=object)
            if any(info.get('session_id') is not None for info in infos):
                batch.non_tensor_batch['session_id'] = np.array([info.get('session_id') for info in infos], dtype=object)
                batch.non_tensor_batch['webshop_task_id'] = np.array(
                    [info.get('webshop_task_id') or info.get('task_key') for info in infos],
                    dtype=object,
                )
                batch.non_tensor_batch['task_key'] = np.array(
                    [info.get('task_key') or info.get('session_id') for info in infos],
                    dtype=object,
                )
            if any(info.get('webshop_state_before') is not None for info in infos):
                batch.non_tensor_batch['webshop_state'] = np.array(
                    [info.get('webshop_state_before') for info in infos],
                    dtype=object,
                )
                batch.non_tensor_batch['webshop_state_after'] = np.array(
                    [info.get('webshop_state_after') for info in infos],
                    dtype=object,
                )
            next_obs_texts = next_obs.get("text", None)
            batch.non_tensor_batch['vmpr_next_observation'] = np.array(
                [
                    extract_current_observation(str(next_obs_texts[i])) or str(next_obs_texts[i])
                    if next_obs_texts is not None
                    else ""
                    for i in range(batch_size)
                ],
                dtype=object,
            )

            if state_histories is not None:
                for i in range(batch_size):
                    if not active_masks[i]:
                        continue
                    action = str(infos[i].get("executed_action", "") or "").strip()
                    if not action:
                        continue
                    valid = self._vmpr_valid_action_flag(infos[i].get("is_action_valid", True))
                    clean = self._vmpr_is_clean_action(action)
                    if clean:
                        next_obs_text = str(next_obs_texts[i] if next_obs_texts is not None else "")
                        next_current_observation = extract_current_observation(next_obs_text) or next_obs_text
                        next_location = infer_location_from_observation(next_current_observation)
                        if next_location != "unknown":
                            state_histories[i]["last_location"] = next_location
                        if not valid:
                            state_histories[i]["no_progress_actions"].append(f"turn {_step}: {compact_action_text(action)} -> not admissible")
                            state_histories[i]["last_invalid_action"] = f"turn {_step}: {compact_action_text(action)} -> not admissible"
                        elif "nothing happens" in normalize_observation_text(next_current_observation):
                            state_histories[i]["no_progress_actions"].append(f"turn {_step}: {compact_action_text(action)} -> nothing happened")
                            state_histories[i]["last_invalid_action"] = f"turn {_step}: {compact_action_text(action)} -> nothing happened"
                        else:
                            state_histories[i]["valid_actions"].append(action)
                            state_histories[i]["last_invalid_action"] = ""
                    else:
                        state_histories[i]["malformed_actions"].append(f"turn {_step}: {compact_action_text(action)}")
                    for key in ("valid_actions", "no_progress_actions", "malformed_actions"):
                        if len(state_histories[i][key]) > 64:
                            state_histories[i][key] = state_histories[i][key][-64:]

            if 'tool_calling' in infos[0]:
                tool_callings[active_masks] += np.array([info['tool_calling'] for info in infos], dtype=np.float32)[active_masks]
            # Create reward tensor, only assign rewards for active environments
            # episode_rewards += torch_to_numpy(rewards) * torch_to_numpy(active_masks)
            episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
            episode_lengths[active_masks] += 1

            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)
            
            # Update episode lengths for active environments
            batch_list: list[dict] = to_list_of_dict(batch)

            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])

            # Update done states
            is_done = np.logical_or(is_done, dones)
                
            # Update observations for next step
            obs = next_obs

            # Break if all environments are done
            if is_done.all():
                break
        
        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards, 
                    episode_lengths=episode_lengths,
                    )
        self._last_total_infos = total_infos
        
        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings
    
    def dynamic_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Conduct dynamic rollouts until a target batch size is met. 
        Keeps sampling until the desired number of effective trajectories is collected.
        Adopted from DAPO (https://arxiv.org/abs/2503.14476)

        Args:
            gen_batch (DataProto): Initial batch for rollout.
            actor_rollout_wg: Actor model workers for generating responses.
            envs (EnvironmentManagerBase): Environment manager instance.

        Returns:
            total_batch_list (List[Dict]): Complete set of rollout steps.
            total_episode_rewards (np.ndarray): Accumulated rewards.
            total_episode_lengths (np.ndarray): Lengths per episode.
            total_success (Dict[str, np.ndarray]): Success metrics.
            total_traj_uid (np.ndarray): Trajectory IDs.
        """
        total_batch_list = []
        total_episode_rewards = []
        total_episode_lengths = []
        total_success = []
        total_traj_uid = []
        total_tool_callings = []
        try_count: int = 0
        max_try_count = self.config.algorithm.filter_groups.max_num_gen_batches

        while len(total_batch_list) < self.config.data.train_batch_size * self.config.env.rollout.n and try_count < max_try_count:

            if len(total_batch_list) > 0:
                print(f"valid num={len(total_batch_list)} < target num={self.config.data.train_batch_size * self.config.env.rollout.n}. Keep generating... ({try_count}/{max_try_count})")
            try_count += 1

            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = filter_group_data(batch_list=batch_list, 
                                                                                                episode_rewards=episode_rewards, 
                                                                                                episode_lengths=episode_lengths, 
                                                                                                success=success, 
                                                                                                traj_uid=traj_uid, 
                                                                                                tool_callings=tool_callings, 
                                                                                                config=self.config,
                                                                                                last_try=(try_count == max_try_count),
                                                                                                )
            
            total_batch_list += batch_list
            total_episode_rewards.append(episode_rewards)
            total_episode_lengths.append(episode_lengths)
            total_success.append(success)
            total_traj_uid.append(traj_uid)
            total_tool_callings.append(tool_callings)

        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        total_success = {key: np.concatenate([success[key] for success in total_success], axis=0) for key in total_success[0].keys()}
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)
        total_tool_callings = np.concatenate(total_tool_callings, axis=0)

        return total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_tool_callings

    def multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            global_step: int = 0,
            ) -> DataProto:
        """
        Select and run the appropriate rollout loop (dynamic or vanilla).

        Args:
            gen_batch (DataProto): Initial prompt batch.
            actor_rollout_wg: Actor model workers.
            envs (EnvironmentManagerBase): Environment manager for interaction.
            is_train (bool): Whether in training mode (affects dynamic sampling).
            global_step (int): Trainer step used for VMPR prefix freshness and persistence.

        Returns:
            DataProto: Final collected trajectory data with metadata.
        """
        if is_train:
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)
        gen_batch = self._vmpr_prepare_gen_batch(gen_batch, global_step=global_step, is_train=is_train)

        actor_rollout_ref_cfg = getattr(self.config, "actor_rollout_ref", None)
        rollout_cfg = getattr(actor_rollout_ref_cfg, "rollout", None) if actor_rollout_ref_cfg is not None else None
        keep_engine_awake = bool(rollout_cfg.get("keep_engine_awake_during_multiturn", False)) if rollout_cfg is not None else False
        if keep_engine_awake and (
            not hasattr(actor_rollout_wg, "begin_rollout_generation")
            or not hasattr(actor_rollout_wg, "end_rollout_generation")
        ):
            raise RuntimeError(
                "keep_engine_awake_during_multiturn requires rollout workers with "
                "begin_rollout_generation/end_rollout_generation support"
            )

        try:
            if keep_engine_awake:
                actor_rollout_wg.begin_rollout_generation()

            # Initial observations from the environment
            if self.config.algorithm.filter_groups.enable and is_train:
                # Dynamic Sampling (for DAPO and Dynamic GiGPO)
                total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                    self.dynamic_multi_turn_loop(
                    gen_batch=gen_batch,
                    actor_rollout_wg=actor_rollout_wg,
                    envs=envs,
                )
            else:
                # Vanilla Sampling
                total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                    self.vanilla_multi_turn_loop(
                    gen_batch=gen_batch,
                    actor_rollout_wg=actor_rollout_wg,
                    envs=envs,
                )
        finally:
            if keep_engine_awake:
                actor_rollout_wg.end_rollout_generation()
        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)
        assert len(total_batch_list) == len(totoal_tool_callings)

        self._vmpr_update_prefix_buffer(
            total_batch_list=total_batch_list,
            total_infos=getattr(self, "_last_total_infos", []),
            episode_rewards=total_episode_rewards,
            success=total_success,
            global_step=global_step,
            is_train=is_train,
        )
        

        # Create trajectory data
        gen_batch_output: DataProto = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
            tool_callings=totoal_tool_callings,
        )
        gen_batch_output.meta_info["vmpr_metrics"] = dict(self._vmpr_pending_metrics)
        
        return gen_batch_output
