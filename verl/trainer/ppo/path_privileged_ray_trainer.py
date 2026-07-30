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

"""Trainer for path-privileged online preference distillation."""

import torch
import numpy as np

from verl import DataProto
from verl.trainer.ppo.path_privileged_utils import (
    PathPrivilegedContextProvider,
    build_path_candidate_ce_batch,
    build_path_privileged_teacher_batch,
    extract_current_observation,
    is_clean_executed_action,
    is_truthy,
    normalize_observation_text,
)
from verl.trainer.ppo.rlsd_utils import SkillProvider
from verl.trainer.ppo.skillsd_ray_trainer import SkillSDRayTrainer


class PathPrivOPDRayTrainer(SkillSDRayTrainer):
    """SkillSD-compatible trainer with GT-path privileged teacher prompts."""

    def __init__(self, *args, skill_provider: SkillProvider = None, path_context_provider: PathPrivilegedContextProvider = None, **kwargs):
        super().__init__(*args, skill_provider=skill_provider, **kwargs)
        self.path_context_provider = path_context_provider or PathPrivilegedContextProvider(config=self.config, skill_provider=skill_provider)

    @staticmethod
    def _as_py(value):
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return value.item()
            return value.tolist()
        if hasattr(value, "item") and not isinstance(value, (str, bytes, dict, list, tuple)):
            try:
                return value.item()
            except Exception:
                return value
        return value

    def _maybe_apply_in_batch_success_paths(self, batch: DataProto) -> dict:
        path_cfg = self.config.algorithm.get("path_opd", {}) if hasattr(self.config, "algorithm") else {}
        if not bool(path_cfg.get("in_batch_success_path", False)):
            return {}
        filter_no_progress_success_actions = bool(path_cfg.get("filter_no_progress_success_actions", True))

        traj_values = batch.non_tensor_batch.get("traj_uid")
        action_values = batch.non_tensor_batch.get("executed_action")
        turn_values = batch.non_tensor_batch.get("turn_step")
        valid_values = batch.non_tensor_batch.get("is_action_valid")
        if traj_values is None or action_values is None or turn_values is None:
            return {"path_opd/in_batch_success_path_enabled": 1.0, "path_opd/in_batch_success_path_available_ratio": 0.0}

        batch_size = len(traj_values)
        uid_values = batch.non_tensor_batch.get("uid")
        if uid_values is None:
            uid_values = batch.non_tensor_batch.get("gamefile")
        if uid_values is None:
            uid_values = np.array([str(self._as_py(traj_values[i])) for i in range(batch_size)], dtype=object)

        reward_values = batch.non_tensor_batch.get("episode_rewards")
        if reward_values is None:
            reward_values = batch.non_tensor_batch.get("score")
        if reward_values is None:
            reward_values = np.zeros(batch_size, dtype=object)

        grouped: dict[str, dict[str, dict[int, str]]] = {}
        observations_by_traj: dict[str, dict[int, str]] = {}
        traj_success: dict[str, bool] = {}
        traj_group: dict[str, str] = {}
        dropped_actions = 0
        duplicate_turns = 0
        no_progress_actions = 0
        input_values = batch.non_tensor_batch.get("input")
        for idx in range(batch_size):
            traj_uid = str(self._as_py(traj_values[idx]) or "")
            if not traj_uid:
                continue
            group_uid = str(self._as_py(uid_values[idx]) or traj_uid)
            turn_step = int(self._as_py(turn_values[idx]) or 0)
            if input_values is not None:
                prompt_text = str(self._as_py(input_values[idx]) or "")
                observations_by_traj.setdefault(traj_uid, {}).setdefault(turn_step, extract_current_observation(prompt_text) or prompt_text)
            action = str(self._as_py(action_values[idx]) or "").strip()
            turn_actions = grouped.setdefault(group_uid, {}).setdefault(traj_uid, {})
            valid_action = is_truthy(valid_values[idx], default=True) if valid_values is not None else True
            if action:
                if not valid_action or not is_clean_executed_action(action):
                    dropped_actions += 1
                elif turn_step not in turn_actions:
                    turn_actions[turn_step] = action
                else:
                    duplicate_turns += 1
            traj_group[traj_uid] = group_uid
            try:
                reward = float(self._as_py(reward_values[idx]) or 0.0)
            except Exception:
                reward = 0.0
            traj_success[traj_uid] = traj_success.get(traj_uid, False) or reward > 0

        group_success_path: dict[str, tuple[str, list[str]]] = {}
        for group_uid, trajectories in grouped.items():
            successful = []
            for traj_uid, turn_actions in trajectories.items():
                if not traj_success.get(traj_uid, False):
                    continue
                ordered = []
                traj_observations = observations_by_traj.get(traj_uid, {})
                for turn, action in sorted(turn_actions.items(), key=lambda item: item[0]):
                    next_observation = traj_observations.get(int(turn) + 1, "")
                    if filter_no_progress_success_actions and next_observation and "nothing happens" in normalize_observation_text(next_observation):
                        no_progress_actions += 1
                        continue
                    ordered.append(action)
                if ordered:
                    successful.append((len(ordered), traj_uid, ordered))
            if successful:
                successful.sort(key=lambda item: (item[0], item[1]))
                group_success_path[group_uid] = (str(successful[0][1]), list(successful[0][2]))

        replace_existing = bool(path_cfg.get("in_batch_success_replace_existing", False))
        metadata_values = batch.non_tensor_batch.get("vmpr_metadata")
        if metadata_values is None:
            metadata_values = np.array([{} for _ in range(batch_size)], dtype=object)
        else:
            metadata_values = np.array(list(metadata_values), dtype=object)

        assigned = 0
        preserved_existing = 0
        for idx in range(batch_size):
            traj_uid = str(self._as_py(traj_values[idx]) or "")
            group_uid = str(self._as_py(uid_values[idx]) or traj_group.get(traj_uid, traj_uid))
            success_entry = group_success_path.get(group_uid)
            if not success_entry:
                continue
            source_traj_uid, actions = success_entry
            metadata = self._as_py(metadata_values[idx])
            metadata = dict(metadata) if isinstance(metadata, dict) else {}
            if metadata.get("full_actions") and not replace_existing:
                preserved_existing += 1
                continue
            metadata["full_actions"] = list(actions)
            metadata["path_source"] = "in_batch_success"
            metadata["in_batch_success_group_uid"] = group_uid
            metadata["source_traj_uid"] = source_traj_uid
            metadata_values[idx] = metadata
            assigned += 1

        batch.non_tensor_batch["vmpr_metadata"] = metadata_values
        return {
            "path_opd/in_batch_success_path_enabled": 1.0,
            "path_opd/in_batch_success_path_group_ratio": len(group_success_path) / max(1.0, len(grouped)),
            "path_opd/in_batch_success_path_available_ratio": assigned / max(1.0, batch_size),
            "path_opd/in_batch_success_path_preserved_existing_ratio": preserved_existing / max(1.0, batch_size),
            "path_opd/in_batch_success_path_dropped_action_ratio": dropped_actions / max(1.0, batch_size),
            "path_opd/in_batch_success_path_duplicate_turn_ratio": duplicate_turns / max(1.0, batch_size),
            "path_opd/in_batch_success_path_no_progress_action_ratio": no_progress_actions / max(1.0, batch_size),
        }

    def _compute_teacher_log_probs(self, batch: DataProto) -> torch.Tensor:
        self._maybe_apply_in_batch_success_paths(batch)
        self._maybe_add_sdl_action_mask(batch)
        self._maybe_add_sdl_special_token_mask(batch)
        teacher_batch = build_path_privileged_teacher_batch(
            batch=batch,
            context_provider=self.path_context_provider,
            tokenizer=self.tokenizer,
            max_prompt_length=self.config.data.max_prompt_length,
            global_step=self.global_steps,
        )
        # The privileged teacher path only consumes token log-probabilities for
        # SDL. Entropy is useful for the normal old_log_prob metric, but here it
        # is discarded and can make the extra teacher forward noticeably slower.
        teacher_batch.meta_info["calculate_entropy"] = False
        path_cfg = self.config.algorithm.get("path_opd", {})
        sdl_loss_mode = str(path_cfg.get("sdl_loss_mode", "chosen_token_k3") or "chosen_token_k3").lower()
        use_topk_sdl = sdl_loss_mode in (
            "topk",
            "topk_forward_kl",
            "teacher_topk_forward_kl",
            "forward_kl_topk",
        )
        if use_topk_sdl:
            topk = int(path_cfg.get("sdl_topk", 32))
            if topk <= 0:
                raise ValueError(f"algorithm.path_opd.sdl_topk must be positive, got {topk}")
            teacher_batch.meta_info["return_topk"] = topk
        self._last_teacher_skill_metrics = dict(teacher_batch.meta_info.get("teacher_skill_metrics", {}))
        sdl_sample_weight = teacher_batch.meta_info.get("sdl_sample_weight")
        if sdl_sample_weight is not None:
            batch.batch["sdl_sample_weight"] = sdl_sample_weight.to(device=batch.batch["responses"].device)
        policy_loss_sample_weight = teacher_batch.meta_info.get("policy_loss_sample_weight")
        policy_loss_filter = str(getattr(self.path_context_provider, "policy_loss_filter", "none") or "none").lower()
        if policy_loss_sample_weight is not None and policy_loss_filter != "none":
            policy_loss_sample_weight = policy_loss_sample_weight.to(device=batch.batch["responses"].device)
            batch.batch["policy_loss_sample_weight"] = policy_loss_sample_weight
        teacher_output = self.actor_rollout_wg.compute_log_prob(teacher_batch)
        teacher_log_probs = teacher_output.batch["old_log_probs"]
        if use_topk_sdl:
            required = ("teacher_topk_ids", "teacher_topk_log_probs")
            missing = [name for name in required if name not in teacher_output.batch]
            if missing:
                raise KeyError(f"Teacher Top-k forward is missing outputs: {missing}")
            teacher_topk_ids = teacher_output.batch["teacher_topk_ids"]
            teacher_topk_log_probs = teacher_output.batch["teacher_topk_log_probs"]
            if teacher_topk_ids.shape[:2] != batch.batch["responses"].shape:
                raise ValueError(
                    "Teacher Top-k response shape mismatch: "
                    f"{tuple(teacher_topk_ids.shape)} vs {tuple(batch.batch['responses'].shape)}"
                )
            batch.batch["teacher_topk_ids"] = teacher_topk_ids
            batch.batch["teacher_topk_log_probs"] = teacher_topk_log_probs
        return teacher_log_probs

    def _prepare_candidate_ce_batch(self, batch: DataProto) -> None:
        """Attach matcher-selected action-only CE targets without a teacher pass."""
        self._maybe_apply_in_batch_success_paths(batch)
        path_cfg = self.config.algorithm.get("path_opd", {})
        candidate_batch = build_path_candidate_ce_batch(
            batch=batch,
            context_provider=self.path_context_provider,
            tokenizer=self.tokenizer,
            max_response_length=int(path_cfg.get("candidate_ce_max_response_length", 64)),
            global_step=self.global_steps,
        )
        device = batch.batch["responses"].device
        candidate_key_map = {
            "input_ids": "candidate_ce_input_ids",
            "attention_mask": "candidate_ce_attention_mask",
            "position_ids": "candidate_ce_position_ids",
            "responses": "candidate_ce_responses",
            "candidate_response_mask": "candidate_ce_response_mask",
            "candidate_sample_weight": "candidate_ce_sample_weight",
            "normalization_token_count": "candidate_ce_normalization_token_count",
        }
        for source_key, destination_key in candidate_key_map.items():
            batch.batch[destination_key] = candidate_batch.batch[source_key].to(device=device)

        policy_loss_sample_weight = candidate_batch.meta_info.get("policy_loss_sample_weight")
        policy_loss_filter = str(getattr(self.path_context_provider, "policy_loss_filter", "none") or "none").lower()
        if policy_loss_sample_weight is not None and policy_loss_filter != "none":
            policy_loss_sample_weight = policy_loss_sample_weight.to(device=device)
            batch.batch["policy_loss_sample_weight"] = policy_loss_sample_weight

        self._last_teacher_skill_metrics = dict(candidate_batch.meta_info.get("candidate_ce_metrics", {}))

    @staticmethod
    def _compute_teacher_topk_metrics(
        batch: DataProto,
        teacher_topk_ids: torch.Tensor,
        teacher_topk_log_probs: torch.Tensor,
    ) -> dict:
        return {}

    def _record_policy_loss_sample_weight_metrics(self, batch: DataProto, policy_loss_sample_weight: torch.Tensor) -> None:
        """Record D12 coverage; the actor applies the weight to its real loss mask."""
        if "responses" not in batch.batch or "response_mask" not in batch.batch:
            return
        if policy_loss_sample_weight.numel() != batch.batch["responses"].shape[0]:
            raise ValueError(
                f"policy_loss_sample_weight batch size mismatch: got {policy_loss_sample_weight.numel()} "
                f"for batch size {batch.batch['responses'].shape[0]}"
            )

        response_mask = batch.batch["response_mask"].to(dtype=torch.float32)
        sample_weight = policy_loss_sample_weight.to(dtype=torch.float32, device=response_mask.device).clamp(min=0.0)
        weighted_response_mask = response_mask * sample_weight.unsqueeze(-1)

        total_response_tokens = response_mask.float().sum().clamp(min=1.0)
        kept_response_tokens = weighted_response_mask.float().sum()
        self._last_teacher_skill_metrics["path_opd/policy_loss_token_ratio"] = (kept_response_tokens / total_response_tokens).detach().item()
        self._last_teacher_skill_metrics["path_opd/policy_loss_sample_weight_mean"] = sample_weight.float().mean().detach().item()
        self._last_teacher_skill_metrics["path_opd/policy_loss_sample_weight_nonzero_ratio"] = (sample_weight > 0).float().mean().detach().item()

    def _maybe_add_sdl_special_token_mask(self, batch: DataProto) -> dict:
        path_cfg = self.config.algorithm.get("path_opd", {}) if hasattr(self.config, "algorithm") else {}
        if not bool(path_cfg.get("sdl_mask_special_tokens", False)):
            return {}
        if "responses" not in batch.batch or "response_mask" not in batch.batch:
            return {}

        special_ids = set(int(token_id) for token_id in getattr(self.tokenizer, "all_special_ids", []) if token_id is not None)
        if not special_ids:
            return {}

        responses = batch.batch["responses"]
        response_mask = batch.batch["response_mask"].bool()
        special_ids_tensor = torch.tensor(sorted(special_ids), device=responses.device, dtype=responses.dtype)
        special_token_mask = torch.isin(responses, special_ids_tensor) & response_mask
        keep_mask = response_mask & ~special_token_mask
        batch.batch["sdl_special_token_keep_mask"] = keep_mask.to(dtype=batch.batch["response_mask"].dtype)

        return {}

    @staticmethod
    def _action_spans(text: str) -> list[tuple[int, int]]:
        lower = text.lower()
        spans: list[tuple[int, int]] = []
        pos = 0
        while True:
            start = lower.find("<action", pos)
            if start < 0:
                break
            tag_end = lower.find(">", start)
            close = lower.find("</action>", tag_end + 1 if tag_end >= 0 else start)
            if tag_end < 0 or close < 0:
                pos = start + 1
                continue
            end = close + len("</action>")
            spans.append((start, end))
            pos = end
        return spans

    def _maybe_add_sdl_action_mask(self, batch: DataProto) -> dict:
        path_cfg = self.config.algorithm.get("path_opd", {}) if hasattr(self.config, "algorithm") else {}
        token_scope = str(path_cfg.get("sdl_token_scope", "all") or "all").lower()
        if token_scope not in {"action", "action_only", "action_span"}:
            return {}
        if "responses" not in batch.batch or "response_mask" not in batch.batch:
            return {}

        responses = batch.batch["responses"]
        response_mask = batch.batch["response_mask"].bool()
        action_mask = torch.zeros_like(response_mask, dtype=response_mask.dtype)

        for row_idx in range(responses.size(0)):
            valid_positions = response_mask[row_idx].nonzero(as_tuple=True)[0]
            if valid_positions.numel() <= 0:
                continue
            token_ids = responses[row_idx][valid_positions].detach().cpu().tolist()
            token_texts = [self.tokenizer.decode([int(token_id)], skip_special_tokens=False) for token_id in token_ids]
            decoded = "".join(token_texts)
            spans = self._action_spans(decoded)
            if not spans:
                continue
            char_start = 0
            for token_idx, token_text in enumerate(token_texts):
                char_end = char_start + len(token_text)
                if any(char_start < span_end and char_end > span_start for span_start, span_end in spans):
                    action_mask[row_idx, valid_positions[token_idx]] = True
                char_start = char_end

        batch.batch["sdl_action_mask"] = action_mask.to(dtype=batch.batch["response_mask"].dtype)
        return {}
