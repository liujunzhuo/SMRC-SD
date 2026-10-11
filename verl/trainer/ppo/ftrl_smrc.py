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

"""FTRL SMRC teacher context and episode-level execution reward integration."""

import json
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch

from agent_system.environments.env_package.ftrl.core import information_atom, match_reference
from agent_system.environments.env_package.ftrl.query_equivalence import load_query_equivalence, task_query_fields
from agent_system.environments.env_package.ftrl.state_defaults import load_state_defaults, task_defaults
from agent_system.multi_turn_rollout.ftrl import prompt_row
from verl import DataProto
from verl.trainer.ppo.path_privileged_ray_trainer import PathPrivOPDRayTrainer
from verl.trainer.ppo.path_privileged_utils import insert_privileged_context
from verl.utils.model import compute_position_id_with_mask


class FTRLContextProvider:
    policy_loss_filter = "none"

    def __init__(self, config):
        self.config = config
        self.state_defaults = load_state_defaults(config.env.ftrl.get("state_defaults"))
        self.query_equivalence = load_query_equivalence(config.env.ftrl.get("query_equivalence"), self.state_defaults)
        with open(config.env.ftrl.references) as handle:
            self.references = {x["task_id"]: x for line in handle if (x := json.loads(line))["valid"]}
        for key, ref in self.references.items():
            expected_mode = self.state_defaults["state_mode"] if self.state_defaults else "raw_query"
            if ref.get("state_mode", "raw_query") != expected_mode:
                raise ValueError("Reference/collector state modes differ")
            if self.state_defaults:
                task_defaults(self.state_defaults, key)
                if ref["state_defaults_sha256"] != self.state_defaults["sha256"]:
                    raise ValueError("Reference/default artifact hash mismatch")
            expected_query_hash = self.query_equivalence["sha256"] if self.query_equivalence else None
            if ref.get("query_equivalence_sha256") != expected_query_hash:
                raise ValueError("Reference/query-equivalence artifact hash mismatch")
            task_query_fields(self.query_equivalence, key)
        self.trajectory_index = SimpleNamespace(by_key=self.references)

    def build_teacher_batch(self, batch, tokenizer, global_step):
        variant = self.config.env.ftrl.variant
        mode = "subset" if variant == "subset" else "exact"
        teacher_view = self.config.env.ftrl.get("teacher_view", "auto")
        if teacher_view == "auto":
            teacher_view = "plain_fullpath" if variant == "fullpath" else "state_candidate"
        if teacher_view not in {"plain_fullpath", "state_candidate"}:
            raise ValueError("Unknown FTRL teacher view")
        contexts, sample_metrics, weights, tensors = [], [], [], []
        for i in range(len(batch)):
            key = batch.non_tensor_batch["task_key"][i]
            reference = self.references.get(key)
            atoms = batch.non_tensor_batch["ftrl_atoms"][i]
            match = match_reference(reference, atoms, mode) if reference else {"matched": False, "kind": "missing_reference", "index": -1, "candidate": "", "anchor_atoms": []}
            selected = reference is not None and (variant == "fullpath" or match["matched"])
            context = ""
            if reference:
                context = "[Reference trajectory]\n" + reference["full_path"] + "\n[/Reference trajectory]"
                if teacher_view == "state_candidate":
                    observed = {}
                    for event in batch.non_tensor_batch["ftrl_history"][i]:
                        for call, output in zip(event["calls"], event["outputs"], strict=True):
                            atom = information_atom(output, call, task_defaults(self.state_defaults, key), task_query_fields(self.query_equivalence, key))
                            if atom is not None:
                                observed[atom] = {"query": call, "observed_result": output}
                    context += "\n[Current observed information state]\n" + json.dumps({"acquired_information": [observed[a] for a in sorted(observed)], "matched_reference_step": match["index"]}, ensure_ascii=False)
                    if match["matched"]:
                        context += "\n[Compatible next action]\n" + match["candidate"] + "\nUse the reference as guidance while responding to the actual current interaction."
            response_length = batch.batch["responses"].shape[-1]
            original = batch.batch["prompts"][i]
            mask = batch.batch["attention_mask"][i, :-response_length].bool()
            text = tokenizer.decode(original[mask], skip_special_tokens=False)
            ids = tokenizer.encode(insert_privileged_context(text, context), add_special_tokens=False)
            overflow = len(ids) > self.config.env.ftrl.teacher_prompt_length
            if overflow:
                ids = original[mask].tolist()
                selected = False
            row = prompt_row(ids, tokenizer, self.config.env.ftrl.teacher_prompt_length)
            full_ids = torch.cat((row["input_ids"], batch.batch["responses"][i]))
            full_mask = torch.cat((row["attention_mask"], batch.batch["attention_mask"][i, -response_length:]))
            tensors.append((full_ids, full_mask, compute_position_id_with_mask(full_mask[None])[0]))
            contexts.append(context)
            weights.append(float(selected))
            sample_metrics.append(
                {
                    "path_opd/recovery_state_match_ratio": float(match["matched"]),
                    "path_opd/sdl_sample_weight_mean": float(selected),
                    "ftrl/reference_available": float(reference is not None),
                    "ftrl/teacher_context_overflow": float(overflow),
                    "ftrl/initial_turn": float(batch.non_tensor_batch["turn_step"][i] == 0),
                    "ftrl/matched_step": match["index"],
                    "ftrl/observed_atoms": len(atoms),
                    "ftrl/completion_match": float(match["kind"] == "completion"),
                }
            )
        batch.non_tensor_batch["path_privileged_context"] = np.array(contexts, dtype=object)
        batch.non_tensor_batch["path_opd_sample_metrics"] = np.array(sample_metrics, dtype=object)
        metrics = {name: float(np.mean([row[name] for row in sample_metrics])) for name in sample_metrics[0]}
        later = [x for x in sample_metrics if not x["ftrl/initial_turn"]]
        metrics["ftrl/match_ratio_after_initial"] = float(np.mean([x["path_opd/recovery_state_match_ratio"] for x in later])) if later else 0.0
        for source in sorted(set(batch.non_tensor_batch["data_source"])):
            indices = [i for i, x in enumerate(batch.non_tensor_batch["data_source"]) if x == source and not sample_metrics[i]["ftrl/initial_turn"]]
            metrics["ftrl/match_after_initial/" + source.split("/")[-1]] = float(np.mean([sample_metrics[i]["path_opd/recovery_state_match_ratio"] for i in indices])) if indices else 0.0
        metrics["path_opd/sdl_sample_weight_nonzero_ratio"] = float(np.mean(weights))
        return DataProto.from_dict(
            tensors={"input_ids": torch.stack([x[0] for x in tensors]), "attention_mask": torch.stack([x[1] for x in tensors]), "position_ids": torch.stack([x[2] for x in tensors]), "responses": batch.batch["responses"]},
            meta_info={"teacher_skill_metrics": metrics, "sdl_sample_weight": torch.tensor(weights), "policy_loss_sample_weight": torch.ones(len(batch))},
        )


class FTRLRewardManager:
    def __init__(self, **kwargs):
        pass

    def __call__(self, batch, return_dict=False):
        rewards = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
        lengths = batch.batch["attention_mask"][:, -rewards.shape[1] :].sum(-1).long()
        values = torch.tensor(batch.non_tensor_batch["episode_rewards"].astype(float), dtype=torch.float32)
        rewards[torch.arange(len(batch)), lengths - 1] = values
        extras = {k: [x[k] for x in batch.non_tensor_batch["ftrl_scores"]] for k in ("solve_precision", "solve_rate", "solve_f1")}
        return {"reward_tensor": rewards, "reward_extra_info": extras} if return_dict else rewards


def selected_signal_metrics(batch, teacher_log_probs):
    """Diagnostic old-policy signals on the actual SDL support; no loss changes."""
    if not all(key in batch.batch for key in ("old_log_probs", "response_mask", "sdl_sample_weight")):
        return {}
    result = {}
    with torch.no_grad():
        gap = teacher_log_probs.detach() - batch.batch["old_log_probs"].detach()
        mask = batch.batch["response_mask"].float() * batch.batch["sdl_sample_weight"].float().unsqueeze(-1)
        rows = batch.non_tensor_batch["path_opd_sample_metrics"]
        groups = {
            "selected": [True for _ in rows],
            "initial": [bool(r["ftrl/initial_turn"]) for r in rows],
            "completion": [bool(r["ftrl/completion_match"]) for r in rows],
            "later_tool": [not r["ftrl/initial_turn"] and bool(r["path_opd/recovery_state_match_ratio"]) and not r["ftrl/completion_match"] for r in rows],
            "unmatched": [not r["path_opd/recovery_state_match_ratio"] for r in rows],
        }
        for name, flags in groups.items():
            support = mask * torch.tensor(flags, device=mask.device).unsqueeze(-1)
            count = support.sum()
            prefix = "ftrl/signal/" + name
            result[prefix + "_tokens"] = count.item()
            result[prefix + "_gap_mean"] = ((gap * support).sum() / count.clamp(min=1)).item()
            result[prefix + "_positive_ratio"] = (((gap > 0).float() * support).sum() / count.clamp(min=1)).item()
    return result


class FTRLSMRCTrainer(PathPrivOPDRayTrainer):
    def _compute_path_signal_metrics(self, batch, teacher_log_probs):
        metrics = super()._compute_path_signal_metrics(batch, teacher_log_probs)
        metrics.update(selected_signal_metrics(batch, teacher_log_probs))
        return metrics

    def _validate(self):
        records, inputs, outputs, scores, extras = [], [], [], [], defaultdict(list)
        for number, raw in enumerate(self.val_dataloader):
            batch = DataProto.from_single_dict(raw)
            batch.non_tensor_batch["validation_item_id"] = np.array([f"{number}:{i}" for i in range(len(batch))], dtype=object)
            n = int(self.config.actor_rollout_ref.rollout.val_kwargs.n)
            original_size = len(batch)
            batch = batch.repeat(repeat_times=n, interleave=True)
            batch.non_tensor_batch["validation_sample_index"] = np.tile(np.arange(n), original_size)
            batch.meta_info = {"eos_token_id": self.tokenizer.eos_token_id, "pad_token_id": self.tokenizer.pad_token_id, "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample, "validate": True}
            result = self.traj_collector.multi_turn_loop(batch, self.actor_rollout_wg, is_train=False, global_step=self.global_steps)
            unique = set()
            for i in range(len(result)):
                uid = result.non_tensor_batch["traj_uid"][i]
                if uid not in unique:
                    unique.add(uid)
                    records.append((result.non_tensor_batch["data_source"][i], result.non_tensor_batch["ftrl_scores"][i]))
                inputs.append(self.tokenizer.decode(result.batch["prompts"][i], skip_special_tokens=False))
                outputs.append(self.tokenizer.decode(result.batch["responses"][i], skip_special_tokens=False))
                scores.append(float(result.non_tensor_batch["episode_rewards"][i]))
            for key in ("traj_uid", "task_key", "turn_step", "data_source", "episode_rewards", "episode_lengths", "ftrl_scores", "ftrl_end_reason", "validation_item_id", "validation_sample_index"):
                extras[key].extend(result.non_tensor_batch[key].tolist())
        if self.config.trainer.get("validation_data_dir"):
            self._dump_generations(inputs, outputs, scores, extras, self.config.trainer.validation_data_dir)
        metrics = {}
        for name in ("solve_precision", "solve_rate", "solve_f1"):
            metrics["val/ftrl/" + name] = float(np.mean([x[name] for _, x in records]))
            for source in sorted({s for s, _ in records}):
                metrics["val/" + source + "/" + name] = float(np.mean([x[name] for s, x in records if s == source]))
        metrics["val/ftrl/episodes"] = len(records)
        return metrics

    def _rollout_dump_extra_infos(self, batch, reward_extra_infos_dict):
        result = super()._rollout_dump_extra_infos(batch, reward_extra_infos_dict)
        for key in ("ftrl_atoms", "ftrl_history", "ftrl_scores", "ftrl_end_reason", "episode_rewards", "episode_lengths", "data_source"):
            result[key] = batch.non_tensor_batch[key]
        return result
